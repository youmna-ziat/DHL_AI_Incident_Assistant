"""
llm_local.py
============

Génération de la recommandation finale avec un LLM local.

Le LLM intervient uniquement APRÈS le retrieval / clustering /
sélection algorithmique.

Si les preuves sont insuffisantes ou trop génériques, le LLM est
automatiquement court-circuité et le système utilise la recommandation
heuristique existante.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional


# ================================================================
# IMPORT DU FALLBACK HEURISTIQUE
# ================================================================

try:
    from recommendation_core import generate_natural_recommendation, classify_solution
except ImportError:  # pragma: no cover
    generate_natural_recommendation = None
    classify_solution = None


# ================================================================
# CONFIGURATION DU MODÈLE
# ================================================================

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Permet de changer de modèle sans modifier le code :
#
# PowerShell :
# $env:LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
#
MODEL_NAME = os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)

_pipe_cache: dict = {}


# ================================================================
# 1. GPU
# ================================================================

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ================================================================
# 2. ESPACE DISQUE
# ================================================================

def _check_disk_space(min_gb: float = 2.5) -> None:
    """Vérifie l'espace disque avant téléchargement."""

    free_bytes = shutil.disk_usage(Path.home()).free
    free_gb = free_bytes / (1024 ** 3)

    if free_gb < min_gb:
        raise RuntimeError(
            f"Espace disque insuffisant : {free_gb:.2f} Go libres. "
            f"Il faut au moins environ {min_gb:.1f} Go libres."
        )


def _model_already_cached(model_name: str) -> bool:
    """
    True si le modèle est déjà entièrement présent dans le cache
    HuggingFace local (donc aucun téléchargement ne sera nécessaire).

    Évite de bloquer sur l'espace disque quand ce n'est pas justifié :
    le check original s'appliquait même quand le modèle était déjà
    en cache, alors qu'aucun octet supplémentaire n'allait être écrit.
    """

    cache_root = Path(
        os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ) / "hub"

    repo_folder = "models--" + model_name.replace("/", "--")
    model_path = cache_root / repo_folder
    snapshots = model_path / "snapshots"

    if not snapshots.exists():
        return False

    for snapshot_dir in snapshots.iterdir():
        if any(snapshot_dir.glob("*.safetensors")) or any(
            snapshot_dir.glob("*.bin")
        ):
            return True

    return False


# ================================================================
# 3. CHARGEMENT DU LLM
# ================================================================

def get_llm_pipeline(model_name: str = MODEL_NAME):
    """Charge le modèle une seule fois et le met en cache."""

    if "pipe" not in _pipe_cache:

        from transformers import pipeline

        if not _model_already_cached(model_name):
            _check_disk_space()
        else:
            print(f"Modèle déjà en cache local ({model_name}) -- pas de téléchargement, pas de vérification d'espace disque.")

        print("=" * 70)
        print(f"CHARGEMENT DU LLM LOCAL")
        print(f"Modèle : {model_name}")
        print("=" * 70)

        _pipe_cache["pipe"] = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto" if _has_gpu() else None,
            dtype="auto",
        )

        print("LLM local chargé.")
        print()

    return _pipe_cache["pipe"]


# ================================================================
# 4. SANITIZATION
# ================================================================

_PASSWORD_PATTERN = re.compile(
    r"(?i)(password|mot de passe)\s*[:=]\s*\S+"
)

_LOGIN_PATTERN = re.compile(
    r"(?i)(login|username|user)\s*[:=]\s*\S+"
)


def _sanitize(text: str) -> str:

    text = str(text)

    text = _PASSWORD_PATTERN.sub(
        r"\1: [MASQUÉ]",
        text
    )

    text = _LOGIN_PATTERN.sub(
        r"\1: [MASQUÉ]",
        text
    )

    return text


# ================================================================
# 5. DÉTECTION DES SOLUTIONS PEU INFORMATIVES
# ================================================================

GENERIC_SOLUTIONS = {
    "necessary done",
    "access granted",
    "done",
    "resolved",
    "fixed",
    "completed",
    "closed",
    "issue resolved",
    "problem solved",
    "ok",
    "n/a",
    "na",
    "none",
    "no action",
    "no action required",
    "action completed",
}


def _is_informative_solution(solution: str) -> bool:
    """
    Détermine si une solution historique contient réellement
    une information exploitable pour générer une recommandation.

    Deux critères, dans cet ordre :
      1. Si recommendation_core.classify_solution() la reconnaît
         comme ACTIONABLE (ex: "Password reset.", "Access granted"),
         elle est informative même si elle est courte -- c'est la
         même logique que le reste du pipeline, pas une règle
         dupliquée qui pourrait diverger.
      2. Sinon, on retombe sur les heuristiques génériques
         (texte vide, dans GENERIC_SOLUTIONS, ou trop court).
    """

    if not solution:
        return False

    text = str(solution).strip().lower()

    if not text:
        return False

    if text in GENERIC_SOLUTIONS:
        return False

    if classify_solution is not None:
        if classify_solution(solution) == "ACTIONABLE":
            return True

    # Solutions extrêmement courtes et non reconnues comme actionnables
    if len(text.split()) < 3:
        return False

    return True


def _evidence_quality(evidence: dict) -> tuple[int, int, float]:
    """
    Retourne :
        informative_count
        total_count
        ratio
    """

    tickets = evidence.get("Evidence_Tickets", [])

    if not tickets:
        return 0, 0, 0.0

    informative = 0

    for ticket in tickets:

        solution = ticket.get("Solution", "")

        if _is_informative_solution(solution):
            informative += 1

    total = len(tickets)

    ratio = informative / total if total else 0.0

    return informative, total, ratio


# ================================================================
# 6. GARDE-FOU DE CONFIANCE
# ================================================================

def _llm_should_run(evidence: dict) -> tuple[bool, str]:

    informative, total, ratio = _evidence_quality(evidence)

    # Pas de preuves
    if total == 0:
        return False, "aucune preuve historique disponible"

    # Moins de 50 % des solutions sont informatives
    if ratio < 0.50:
        return (
            False,
            f"preuves insuffisantes : "
            f"{informative}/{total} solutions informatives"
        )

    # Vérification éventuelle du score du cluster
    cluster_score = evidence.get("Cluster_Score")

    if cluster_score is not None:

        try:
            cluster_score = float(cluster_score)

            if cluster_score < 0.30:
                return (
                    False,
                    f"score du cluster trop faible ({cluster_score:.3f})"
                )

        except (ValueError, TypeError):
            pass

    return True, (
        f"preuves suffisantes : "
        f"{informative}/{total} solutions informatives"
    )


# ================================================================
# 7. PROMPT
# ================================================================

SYSTEM_PROMPT = """
Tu es un assistant technique pour l'équipe IT de DHL Maroc.

Tu dois produire une recommandation de résolution à partir
UNIQUEMENT des incidents historiques fournis.

RÈGLES STRICTES :

1. N'invente aucune cause technique.
2. N'invente aucun logiciel, serveur, configuration, procédure
   ou symptôme qui n'est pas présent dans les preuves.
3. Ne transforme pas une hypothèse en fait.
4. Si les preuves ne permettent pas de déterminer précisément
   la cause ou la procédure, indique-le clairement.
5. Utilise uniquement les solutions historiques réellement
   informatives.
6. Ignore les solutions vagues comme "Necessary done",
   "Done", "Resolved", "Access granted", etc.
7. Réponds en français.
8. Donne une recommandation courte et directement exploitable.
9. Maximum 5 étapes.
10. Ne reproduis jamais de mot de passe ou d'identifiant.
11. Ne mentionne JAMAIS le numéro d'un ticket historique
    (ex: INC12345) dans ta réponse -- ces tickets sont des preuves
    internes, pas le sujet de la réponse.
12. Ne recopie pas la description ou la référence d'un incident
    historique. Ne les présente jamais comme s'ils étaient le
    nouveau ticket ou sa résolution.
13. FORMAT DE RÉPONSE STRICT : uniquement une liste à puces de
    2 à 5 actions concrètes à réaliser pour le NOUVEAU ticket.
    Aucun titre, aucune section "Référence", "Description",
    "Résolution du ticket" ou "Conclusion". Pas de phrase
    d'introduction ni de phrase de conclusion. Juste les actions.

IMPORTANT :
Les incidents historiques sont des exemples de résolution,
pas des informations certaines sur le nouveau ticket.
"""


def build_prompt(
    evidence: dict,
    max_evidence_tickets: int = 5
) -> str:

    query = _sanitize(
        evidence.get("Query", "")
    )

    tickets = evidence.get(
        "Evidence_Tickets",
        []
    )[:max_evidence_tickets]

    lines = []

    lines.append(
        "NOUVEAU TICKET :\n"
        + query.strip()
    )

    lines.append(
        "\nINCIDENTS HISTORIQUES PERTINENTS :"
    )

    valid_ticket_count = 0

    for ticket in tickets:

        solution = ticket.get(
            "Solution",
            ""
        )

        if not _is_informative_solution(solution):
            continue

        valid_ticket_count += 1

        desc = _sanitize(
            ticket.get(
                "Short Description",
                ""
            )
        )

        sol = _sanitize(solution)

        sim = ticket.get(
            "Ticket_Similarity"
        )

        if isinstance(sim, (int, float)):
            sim_str = f"{sim:.2f}"
        else:
            sim_str = "?"

        lines.append(
            f"""
{valid_ticket_count}. [{ticket.get("Number", "?")}]
Similarité ticket : {sim_str}

Problème historique :
{desc}

Solution historique :
{sol}
"""
        )

    if valid_ticket_count == 0:

        lines.append(
            "\nAucune solution historique suffisamment "
            "informative n'est disponible."
        )

    lines.append(
        """
TA MISSION :

Propose une recommandation pour le NOUVEAU TICKET ci-dessus
(pas pour les incidents historiques -- eux servent uniquement
de preuve).

Utilise uniquement les informations réellement présentes
dans les solutions historiques ci-dessus.

Si elles ne permettent pas de déterminer une procédure fiable,
dis-le explicitement au lieu d'inventer une solution.

Réponds UNIQUEMENT par une liste à puces des actions à faire.
N'écris ni titre, ni "Référence :", ni "Description :", ni
numéro de ticket historique, ni phrase de conclusion.
"""
    )

    return "\n".join(lines)


_UNWANTED_LINE_PATTERNS = [
    re.compile(r"(?i)^r[ée]solution du ticket\s*:?\s*$"),
    re.compile(r"(?i)^r[ée]f[ée]rence\s*:"),
    re.compile(r"(?i)^description\s*:"),
    re.compile(r"(?i)^conclusion\s*:"),
    re.compile(r"(?i)^ticket\s*:"),
    re.compile(r"(?i)^nouveau ticket\s*:"),
    re.compile(r"(?i)\binc\d{5,}\b"),  # ligne qui mentionne un n° de ticket historique
]


def _clean_llm_output(text: str) -> str:
    """
    Filet de sécurité : retire les lignes qui violent le format
    demandé (titres, "Référence :", numéro de ticket historique,
    "Conclusion :") si le modèle ne suit pas la consigne malgré
    le prompt. Ne touche pas au contenu des actions elles-mêmes.
    """

    cleaned_lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if any(pattern.search(stripped) for pattern in _UNWANTED_LINE_PATTERNS):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    return cleaned if cleaned else text.strip()


# ================================================================
# 8. GÉNÉRATION
# ================================================================

def generate_recommendation_llm(
    evidence: Optional[dict],
    pipe=None,
    max_new_tokens: int = 180,
) -> str:

    # ------------------------------------------------------------
    # Vérification de base
    # ------------------------------------------------------------

    if evidence is None or not isinstance(evidence, dict):

        return (
            "Aucune recommandation fiable ne peut être générée."
        )

    # ------------------------------------------------------------
    # GARDE-FOU
    # ------------------------------------------------------------

    should_run, reason = _llm_should_run(evidence)

    if not should_run:

        print(
            f"[llm_local] LLM non déclenché : {reason} "
            "-> repli heuristique."
        )

        if generate_natural_recommendation is not None:

            return generate_natural_recommendation(
                evidence
            )

        return (
            "Les preuves historiques sont insuffisantes "
            "pour générer une recommandation fiable."
        )

    # ------------------------------------------------------------
    # APPEL LLM
    # ------------------------------------------------------------

    try:

        pipe = pipe or get_llm_pipeline()

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": build_prompt(evidence)
            }
        ]

        prompt_text = (
            pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        output = pipe(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

        text = output[0]["generated_text"].strip()

        if not text:
            raise ValueError(
                "Réponse vide du LLM."
            )

        return _clean_llm_output(text)

    except Exception as error:

        print(
            f"[llm_local] Erreur LLM : {error} "
            "-> repli heuristique."
        )

        if generate_natural_recommendation is not None:

            return generate_natural_recommendation(
                evidence
            )

        return (
            "Aucune recommandation fiable ne peut être générée."
        )