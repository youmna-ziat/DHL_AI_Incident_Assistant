"""
recommendation_core.py
=======================

MODULE UNIQUE ET CENTRAL du pipeline de recommandation d'incidents DHL.

RÈGLE D'OR : toute fonction utilisée par le pipeline (02_pipeline_demo)
ou par le backtest (03_backtest) DOIT être importée depuis ce fichier.
Ne recopiez plus jamais une fonction dans un notebook : si vous devez la
corriger, corrigez-la ICI, une seule fois, puis relancez les notebooks
depuis le début (Kernel > Restart & Run All).

Étapes du pipeline :
    1. load_knowledge_base
    2. add_solution_types
    3. compute_embeddings
    4. search_similar_tickets
    5. filter_actionable_candidates
    6. build_solution_similarity_matrix_*
    7. cluster_similar_solutions
    8. score_solution_clusters
    9. rank_solution_clusters
   10. select_cluster_representatives
   11. select_best_cluster
   12. build_recommendation_evidence
   13. generate_natural_recommendation
   14. recommend_incident
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

REQUIRED_COLUMNS = [
    "Number", "Opened", "Incident Status", "Caller",
    "Assigned Group", "Assignee", "Short Description",
    "Description", "Solution",
]


def load_knowledge_base(
    resolved_path: str,
    escalated_path: str
) -> pd.DataFrame:
    """Charge les 2 fichiers Excel DHL et construit la base de connaissance unique.

    - resolved_path  : tickets traités localement
    - escalated_path : tickets escaladés
    """

    resolved_df = pd.read_excel(resolved_path)
    escalated_df = pd.read_excel(escalated_path)

    resolved_df["source"] = "RESOLVED"
    escalated_df["source"] = "ESCALATED"

    knowledge_base = pd.concat(
        [resolved_df, escalated_df],
        ignore_index=True
    )

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in knowledge_base.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans les fichiers Excel : {missing}"
        )

    # --------------------------------------------------------
    # IMPORTANT :
    # KB_Index est l'identifiant de position stable dans la
    # base complète. Il servira à retrouver les embeddings
    # pré-calculés pendant le backtest.
    # --------------------------------------------------------

    knowledge_base["KB_Index"] = np.arange(len(knowledge_base))

    knowledge_base["text_for_embedding"] = (
        knowledge_base["Short Description"].fillna("").astype(str)
        + " "
        + knowledge_base["Description"].fillna("").astype(str)
    )

    return knowledge_base


# ============================================================
# 2. CLASSIFICATION DES SOLUTIONS
# ============================================================

_ADMINISTRATIVE_PATTERNS = [
    "duplicate",
    "duplicated",
    "close the ticket",
    "ticket to avoid",
    "no response from user",
    "regional ticket opened",
    "new ticket has been raised",
    "a new ticket has been raised",
    "ticket has been raised",
    "ticket opened",
    "data provided",
    "information provided",
    "wrongly routed",
    "cancelled",
    "cancelled by",
]


_NON_ACTIONABLE_PATTERNS = [
    "issue under provider investigations",
    "provider is working on it",
    "provider is working on the issue",
    "no approval is provided",
    "required information needed to resolve",
    "required information was not provided",
    "please ask performance to provide",
    "please reach out to",
    "please contact",
    "please request access from",
    "not our responsibility",
    "no issue from our side",
    "waiting for",
    "waiting on",
    "pending",
    "under investigation",
    "changes ongoing",
    "will be changed",
    "will be configured",
    "will be installed",
]


_GENERIC_SOLUTIONS = {
    "resolved",
    "resolved.",
    "resolved,",
    "solved",
    "solved.",
    "done",
    "done.",
    "fixed",
    "fixed.",
    "closed",
    "closed.",
    "completed",
    "completed.",
    "granted",
    "access granted",
    "access granted.",
    "necessary done",
    "necessary done.",
    "necessaey done.",
    "necessary changes are made.",
    "the necessary changes are made.",
    "change done",
    "change done.",
    "fixed1",
}


_ACTIONABLE_PATTERNS = [
    "installed",
    "installation done",
    "created",
    "was created",
    "has been created",
    "configured",
    "configuration done",
    "updated",
    "has been updated",
    "was updated",
    "changed",
    "has been changed",
    "was changed",
    "replaced",
    "has been replaced",
    "removed",
    "has been removed",
    "reset",
    "password generated",
    "password has been sent",
    "password was sent",
    "unlocked",
    "unlock account",
    "access granted",
    "access provided",
    "granted the",
    "enabled",
    "disabled",
    "recreated",
    "fixed by",
    "released",
    "session released",
    "restored",
    "activated",
    "completed successfully",
    "request completed",
    "changes complete",
    "changes were made",
    "changes are made",
    "set as requested",
    "printer has been created",
    "printers were recreated",
    "rebooted",
    "reconfigured",
    "configured a new",
    "synced",
    "provided the access",
    "has been granted",
]


_RESOLVED_INC_PREFIX = re.compile(
    r"^(resolved|fixed)\s+inc[\w-]+"
)


def classify_solution(solution) -> str:
    """Classe une solution historique."""

    if pd.isna(solution):
        return "NON_ACTIONABLE"

    s = str(solution).strip().lower()

    if not s:
        return "NON_ACTIONABLE"

    if any(
        pattern in s
        for pattern in _ADMINISTRATIVE_PATTERNS
    ):
        return "ADMINISTRATIVE"

    if any(
        pattern in s
        for pattern in _NON_ACTIONABLE_PATTERNS
    ):
        return "NON_ACTIONABLE"

    if (
        s in _GENERIC_SOLUTIONS
        or _RESOLVED_INC_PREFIX.match(s)
    ):
        return "GENERIC"

    if any(
        pattern in s
        for pattern in _ACTIONABLE_PATTERNS
    ):
        return "ACTIONABLE"

    return "NON_ACTIONABLE"


_NON_INFORMATIVE_EXACT = {
    "resolved",
    "resolved.",
    "resolve",
    "solved",
    "solved.",
    "closed",
    "closed.",
    "done",
    "done.",
    "fixed",
    "fixed.",
}


_NON_INFORMATIVE_PATTERNS = [
    "rule of 3",
    "rule of three",
    "set to status 'resolved'",
    "set to status resolved",
    "ticket was set to status",
]


def is_non_informative_solution(solution) -> bool:
    """True si le texte de solution n'apporte aucune information exploitable."""

    if pd.isna(solution):
        return True

    text = str(solution).strip().lower()

    if not text:
        return True

    if text in _NON_INFORMATIVE_EXACT:
        return True

    return any(
        pattern in text
        for pattern in _NON_INFORMATIVE_PATTERNS
    )


def add_solution_types(
    knowledge_base: pd.DataFrame
) -> pd.DataFrame:
    """Ajoute solution_type et non_informative."""

    knowledge_base = knowledge_base.copy()

    knowledge_base["solution_type"] = (
        knowledge_base["Solution"]
        .apply(classify_solution)
    )

    knowledge_base["non_informative"] = (
        knowledge_base["Solution"]
        .apply(is_non_informative_solution)
    )

    return knowledge_base


# ============================================================
# 3. EMBEDDINGS
# ============================================================

# Modèle multilingue conservé volontairement.
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model_cache = {}


def get_model():
    """Charge et met en cache le modèle SentenceTransformer."""

    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache["model"] = SentenceTransformer(
            _MODEL_NAME
        )

    return _model_cache["model"]


def compute_embeddings(
    texts: list[str],
    model=None
) -> np.ndarray:
    """Encode les textes en embeddings normalisés."""

    model = model or get_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings)


# ============================================================
# 4. RECHERCHE DES TICKETS SIMILAIRES
# ============================================================

def search_similar_tickets(
    query: str,
    knowledge_base: pd.DataFrame,
    ticket_embeddings: np.ndarray,
    model=None,
    top_k: int = 20,
    exclude_ticket: Optional[str] = None,
) -> pd.DataFrame:
    """Retourne les top_k tickets historiques les plus proches.

    Parameters
    ----------
    query:
        Nouveau ticket.

    knowledge_base:
        Base historique complète.

    ticket_embeddings:
        Embeddings correspondant exactement aux lignes de
        knowledge_base.

    exclude_ticket:
        Numéro du ticket à exclure.
        Indispensable pour le backtest leave-one-out.
    """

    query_embedding = compute_embeddings(
        [query],
        model=model
    )[0]

    # --------------------------------------------------------
    # Normalisation défensive.
    # Les embeddings sont normalement déjà normalisés par
    # SentenceTransformer, mais on conserve cette étape pour
    # rendre la fonction robuste.
    # --------------------------------------------------------

    norms = np.linalg.norm(
        ticket_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1e-8

    ticket_embeddings_norm = (
        ticket_embeddings / norms
    )

    # Produit scalaire = cosine similarity
    similarities = (
        ticket_embeddings_norm @ query_embedding
    )

    # --------------------------------------------------------
    # Exclusion du ticket testé AVANT le top-k.
    # --------------------------------------------------------

    valid_mask = np.ones(
        len(knowledge_base),
        dtype=bool
    )

    if exclude_ticket is not None:
        valid_mask = (
            knowledge_base["Number"].astype(str).values
            != str(exclude_ticket)
        )

    valid_indices = np.flatnonzero(valid_mask)

    if len(valid_indices) == 0:
        return pd.DataFrame(
            columns=list(knowledge_base.columns)
            + ["Ticket_Similarity"]
        )

    effective_top_k = min(
        top_k,
        len(valid_indices)
    )

    valid_similarities = similarities[valid_indices]

    # --------------------------------------------------------
    # Sélection du top-k uniquement parmi les tickets valides.
    # --------------------------------------------------------

    order = np.argsort(
        valid_similarities
    )[::-1][:effective_top_k]

    top_indices = valid_indices[order]

    candidates = knowledge_base.iloc[
        top_indices
    ].copy()

    candidates["Ticket_Similarity"] = (
        similarities[top_indices]
    )

    # --------------------------------------------------------
    # IMPORTANT :
    # On conserve KB_Index.
    #
    # On peut ensuite reset l'index pandas sans perdre la
    # correspondance avec ticket_embeddings / solution_embeddings.
    # --------------------------------------------------------

    return (
        candidates
        .sort_values(
            "Ticket_Similarity",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 5. FILTRAGE DES SOLUTIONS EXPLOITABLES
# ============================================================

def filter_actionable_candidates(
    candidates: pd.DataFrame
) -> pd.DataFrame:
    """Conserve les solutions réellement exploitables comme preuve.

    Priorité :
        1. ACTIONABLE
        2. si aucune : GENERIC informative

    ADMINISTRATIVE et NON_ACTIONABLE sont écartées.
    """

    candidates = candidates.copy()

    if "non_informative" in candidates.columns:
        usable = candidates[
            ~candidates["non_informative"]
        ].copy()
    else:
        usable = candidates.copy()

    if usable.empty:
        return usable

    actionable = usable[
        usable["solution_type"] == "ACTIONABLE"
    ].copy()

    if not actionable.empty:
        return actionable

    generic = usable[
        usable["solution_type"] == "GENERIC"
    ].copy()

    return generic


# ============================================================
# 6. SIMILARITÉ ENTRE SOLUTIONS CANDIDATES
# ============================================================

def _cosine_similarity_matrix(
    embeddings: np.ndarray
) -> np.ndarray:
    """Calcule une matrice de similarité cosinus sans sklearn."""

    embeddings = np.asarray(
        embeddings,
        dtype=float
    )

    if len(embeddings) == 0:
        return np.empty((0, 0))

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1e-8

    normalized = embeddings / norms

    matrix = normalized @ normalized.T

    # Petite protection contre les erreurs numériques.
    matrix = np.clip(
        matrix,
        -1.0,
        1.0
    )

    return matrix


def build_solution_similarity_matrix_from_texts(
    actionable: pd.DataFrame,
    model=None
) -> pd.DataFrame:
    """Encode les solutions candidates à la volée."""

    if actionable.empty:
        return pd.DataFrame()

    solution_texts = (
        actionable["Solution"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = compute_embeddings(
        solution_texts,
        model=model
    )

    matrix = _cosine_similarity_matrix(
        embeddings
    )

    return pd.DataFrame(
        matrix,
        index=actionable["Number"].tolist(),
        columns=actionable["Number"].tolist(),
    )


def build_solution_similarity_matrix_from_precomputed(
    actionable: pd.DataFrame,
    solution_embeddings: np.ndarray
) -> pd.DataFrame:
    """Utilise les embeddings historiques pré-calculés.

    IMPORTANT :
    on utilise KB_Index et NON l'index pandas courant.

    Cela corrige le bug provoqué par reset_index(drop=True)
    dans search_similar_tickets().
    """

    if actionable.empty:
        return pd.DataFrame()

    if "KB_Index" not in actionable.columns:
        raise ValueError(
            "La colonne 'KB_Index' est absente des candidats. "
            "Elle doit être créée par load_knowledge_base()."
        )

    kb_indices = (
        actionable["KB_Index"]
        .astype(int)
        .to_numpy()
    )

    if np.any(kb_indices < 0):
        raise ValueError(
            "KB_Index contient une valeur négative."
        )

    if np.any(
        kb_indices >= len(solution_embeddings)
    ):
        raise ValueError(
            "KB_Index dépasse la taille de solution_embeddings."
        )

    embeddings = solution_embeddings[
        kb_indices
    ]

    matrix = _cosine_similarity_matrix(
        embeddings
    )

    numbers = actionable["Number"].tolist()

    return pd.DataFrame(
        matrix,
        index=numbers,
        columns=numbers,
    )


# ============================================================
# 7. CLUSTERING DES SOLUTIONS
# ============================================================

def cluster_similar_solutions(
    solution_similarity_matrix: pd.DataFrame,
    threshold: float = 0.75
) -> pd.DataFrame:
    """Regroupe les solutions similaires.

    Implémentation pure NumPy.
    Aucun besoin de sklearn/scipy.

    Règle :
        deux solutions sont reliées si leur similarité >= threshold.

    Les composantes connexes forment les clusters.
    """

    if solution_similarity_matrix.empty:
        return pd.DataFrame(
            columns=["Number", "Cluster"]
        )

    numbers = solution_similarity_matrix.index.tolist()

    n = len(numbers)

    if n == 1:
        return pd.DataFrame({
            "Number": numbers,
            "Cluster": [0],
        })

    matrix = np.asarray(
        solution_similarity_matrix.values,
        dtype=float
    )

    adjacency = (
        matrix >= threshold
    )

    # --------------------------------------------------------
    # Composantes connexes.
    # --------------------------------------------------------

    visited = np.zeros(
        n,
        dtype=bool
    )

    labels = np.full(
        n,
        -1,
        dtype=int
    )

    cluster_id = 0

    for start in range(n):

        if visited[start]:
            continue

        stack = [start]
        visited[start] = True

        while stack:

            current = stack.pop()

            labels[current] = cluster_id

            neighbours = np.flatnonzero(
                adjacency[current]
            )

            for neighbour in neighbours:

                if not visited[neighbour]:

                    visited[neighbour] = True
                    stack.append(
                        int(neighbour)
                    )

        cluster_id += 1

    return pd.DataFrame({
        "Number": numbers,
        "Cluster": labels,
    })


# ============================================================
# 8. SCORING DES CLUSTERS
# ============================================================

def score_solution_clusters(
    actionable: pd.DataFrame,
    clusters: pd.DataFrame,
    solution_similarity_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Calcule les scores descriptifs de chaque cluster."""

    if clusters.empty:
        return pd.DataFrame()

    cluster_results = []

    for cluster_id in sorted(
        clusters["Cluster"].unique()
    ):

        cluster_numbers = (
            clusters.loc[
                clusters["Cluster"] == cluster_id,
                "Number"
            ]
            .tolist()
        )

        cluster_data = actionable[
            actionable["Number"].isin(
                cluster_numbers
            )
        ].copy()

        if cluster_data.empty:
            continue

        ticket_similarity_mean = (
            cluster_data["Ticket_Similarity"].mean()
        )

        ticket_similarity_max = (
            cluster_data["Ticket_Similarity"].max()
        )

        valid_tickets = cluster_data[
            "Number"
        ].tolist()

        if len(valid_tickets) > 1:

            matrix = (
                solution_similarity_matrix
                .loc[
                    valid_tickets,
                    valid_tickets
                ]
                .values
            )

            upper_values = matrix[
                np.triu_indices(
                    len(matrix),
                    k=1
                )
            ]

            solution_coherence = float(
                upper_values.mean()
            )

        else:

            # Valeur neutre conservée.
            solution_coherence = 0.5

        cluster_results.append({
            "Cluster": cluster_id,
            "Number_of_Tickets": len(cluster_data),
            "Ticket_Similarity_Mean": ticket_similarity_mean,
            "Ticket_Similarity_Max": ticket_similarity_max,
            "Solution_Coherence": solution_coherence,
        })

    return pd.DataFrame(
        cluster_results
    )


def rank_solution_clusters(
    cluster_scores: pd.DataFrame
) -> pd.DataFrame:
    """Classe les clusters selon :

        50% pertinence
        30% cohérence
        20% support

    Le support est logarithmique afin de limiter l'effet
    d'un très grand cluster.
    """

    if cluster_scores.empty:
        return cluster_scores.copy()

    scores = cluster_scores.copy()

    max_support = (
        scores["Number_of_Tickets"].max()
    )

    scores["Relevance_Score"] = (
        scores["Ticket_Similarity_Mean"]
    )

    scores["Coherence_Score"] = (
        scores["Solution_Coherence"]
    )

    if max_support <= 1:

        scores["Support_Score"] = 0.0

    else:

        scores["Support_Score"] = np.where(
            scores["Number_of_Tickets"] >= 2,
            np.log1p(
                scores["Number_of_Tickets"]
            ) / np.log1p(max_support),
            0.0,
        )

    scores["Cluster_Final_Score"] = (
        0.50 * scores["Relevance_Score"]
        + 0.30 * scores["Coherence_Score"]
        + 0.20 * scores["Support_Score"]
    )

    return (
        scores
        .sort_values(
            "Cluster_Final_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 9. REPRÉSENTANT ET CLUSTER GAGNANT
# ============================================================

def select_cluster_representatives(
    actionable: pd.DataFrame,
    clusters: pd.DataFrame,
    solution_similarity_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Sélectionne la solution la plus centrale de chaque cluster."""

    if clusters.empty:
        return pd.DataFrame()

    representatives = []

    for cluster_id in sorted(
        clusters["Cluster"].unique()
    ):

        cluster_numbers = (
            clusters.loc[
                clusters["Cluster"] == cluster_id,
                "Number"
            ]
            .tolist()
        )

        cluster_data = actionable[
            actionable["Number"].isin(
                cluster_numbers
            )
        ].copy()

        if cluster_data.empty:
            continue

        if len(cluster_numbers) == 1:

            row = cluster_data.iloc[0]

            representatives.append({
                "Cluster": cluster_id,
                "Representative_Ticket": row["Number"],
                "Representative_Solution": row["Solution"],
                "Ticket_Similarity": row["Ticket_Similarity"],
                "Internal_Similarity": 0.5,
            })

            continue

        cluster_matrix = (
            solution_similarity_matrix
            .loc[
                cluster_numbers,
                cluster_numbers
            ]
        )

        mean_similarities = {
            ticket: cluster_matrix.loc[
                ticket,
                [
                    other
                    for other in cluster_numbers
                    if other != ticket
                ]
            ].mean()
            for ticket in cluster_numbers
        }

        # --------------------------------------------------------
        # IMPORTANT : on ne choisit plus le représentant sur la
        # seule cohérence interne (Internal_Similarity), car cela
        # peut sélectionner une solution "consensuelle" du cluster
        # mais moins pertinente pour CE ticket précis. On combine
        # pertinence (Ticket_Similarity, ce que la requête cherche
        # vraiment) et cohérence (Internal_Similarity, la garantie
        # que la solution n'est pas un cas isolé) à 70/30.
        # --------------------------------------------------------
        ticket_similarity_by_number = {
            row["Number"]: row["Ticket_Similarity"]
            for _, row in cluster_data.iterrows()
        }

        combined_scores = {
            ticket: (
                0.7 * ticket_similarity_by_number[ticket]
                + 0.3 * mean_similarities[ticket]
            )
            for ticket in cluster_numbers
        }

        representative_ticket = max(
            combined_scores,
            key=combined_scores.get
        )

        representative_row = (
            cluster_data[
                cluster_data["Number"]
                == representative_ticket
            ]
            .iloc[0]
        )

        representatives.append({
            "Cluster": cluster_id,
            "Representative_Ticket": representative_ticket,
            "Representative_Solution": representative_row["Solution"],
            "Ticket_Similarity": representative_row["Ticket_Similarity"],
            "Internal_Similarity": mean_similarities[
                representative_ticket
            ],
        })

    return pd.DataFrame(
        representatives
    )


def select_best_cluster(
    ranked_clusters: pd.DataFrame,
    representatives: pd.DataFrame
) -> Optional[dict]:
    """Combine le meilleur cluster avec son représentant."""

    if ranked_clusters.empty:
        return None

    if representatives.empty:
        return None

    best_cluster = ranked_clusters.iloc[0]

    best_cluster_id = best_cluster[
        "Cluster"
    ]

    representative = representatives[
        representatives["Cluster"]
        == best_cluster_id
    ]

    if representative.empty:
        return None

    representative = representative.iloc[0]

    return {
        "Cluster": int(best_cluster_id),
        "Cluster_Final_Score": float(
            best_cluster["Cluster_Final_Score"]
        ),
        "Representative_Ticket": (
            representative["Representative_Ticket"]
        ),
        "Representative_Solution": (
            representative["Representative_Solution"]
        ),
        "Ticket_Similarity": float(
            representative["Ticket_Similarity"]
        ),
        "Internal_Similarity": float(
            representative["Internal_Similarity"]
        ),
        "Number_of_Tickets": int(
            best_cluster["Number_of_Tickets"]
        ),
        "Ticket_Similarity_Mean": float(
            best_cluster["Ticket_Similarity_Mean"]
        ),
        "Solution_Coherence": float(
            best_cluster["Solution_Coherence"]
        ),
    }


# ============================================================
# 10. ÉVIDENCES
# ============================================================

def build_recommendation_evidence(
    query: str,
    best_result: dict,
    clusters: pd.DataFrame,
    actionable: pd.DataFrame,
    top_n_tickets: int = 5,
) -> dict:
    """Construit les preuves justifiant la recommandation."""

    best_cluster_id = best_result[
        "Cluster"
    ]

    cluster_numbers = (
        clusters.loc[
            clusters["Cluster"] == best_cluster_id,
            "Number"
        ]
        .tolist()
    )

    cluster_tickets = actionable[
        actionable["Number"].isin(
            cluster_numbers
        )
    ].copy()

    cluster_tickets = (
        cluster_tickets
        .sort_values(
            "Ticket_Similarity",
            ascending=False
        )
    )

    evidence_tickets = (
        cluster_tickets
        .head(top_n_tickets)[
            [
                "Number",
                "Short Description",
                "Solution",
                "Ticket_Similarity",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    return {
        "Query": query,
        "Winning_Cluster": best_cluster_id,
        "Cluster_Score": best_result[
            "Cluster_Final_Score"
        ],
        "Number_of_Tickets": best_result[
            "Number_of_Tickets"
        ],
        "Ticket_Similarity_Mean": best_result[
            "Ticket_Similarity_Mean"
        ],
        "Solution_Coherence": best_result[
            "Solution_Coherence"
        ],
        "Representative_Ticket": best_result[
            "Representative_Ticket"
        ],
        "Representative_Solution": best_result[
            "Representative_Solution"
        ],
        "Evidence_Tickets": evidence_tickets,
    }


# ============================================================
# 11. GÉNÉRATION DE LA RECOMMANDATION
# ============================================================

def generate_natural_recommendation(
    evidence: Optional[dict]
) -> str:
    """Génère une recommandation en langage naturel.

    Version heuristique de secours.
    Le LLM pourra remplacer cette étape plus tard.
    """

    if (
        evidence is None
        or not isinstance(evidence, dict)
    ):
        return (
            "Aucune recommandation fiable "
            "ne peut être générée."
        )

    representative_solution = str(
        evidence.get(
            "Representative_Solution",
            ""
        )
    ).strip()

    descriptions = [
        str(
            ticket.get(
                "Short Description",
                ""
            )
        )
        for ticket in evidence.get(
            "Evidence_Tickets",
            []
        )
        if ticket.get(
            "Short Description"
        )
    ]

    context_text = (
        representative_solution
        + " "
        + " ".join(descriptions)
    ).lower()

    rules = [

        (
            [
                "printer",
                "printing",
                "print",
                "driver",
            ],
            [
                "Vérifier la configuration de l'imprimante concernée.",
                "Vérifier puis mettre à jour ou réinstaller le pilote de l'imprimante si nécessaire.",
                "Effectuer un test d'impression afin de confirmer la résolution du problème.",
            ],
        ),

        (
            [
                "password",
                "ldap",
                "reset password",
            ],
            [
                "Procéder à la réinitialisation ou à la modification du mot de passe du compte concerné.",
                "Vérifier ensuite que l'utilisateur peut correctement s'authentifier avec ses nouveaux identifiants.",
            ],
        ),

        (
            [
                "account",
                "access",
                "login",
            ],
            [
                "Vérifier les droits et la configuration du compte concerné.",
                "Appliquer les corrections nécessaires sur les paramètres d'accès.",
                "Effectuer un test de connexion afin de confirmer le rétablissement de l'accès.",
            ],
        ),

        (
            [
                "configuration",
                "configured",
                "setup",
                "settings",
            ],
            [
                "Vérifier la configuration du composant concerné.",
                "Appliquer la configuration appropriée puis vérifier le fonctionnement du service.",
            ],
        ),

        (
            [
                "installed",
                "installation",
                "reinstalled",
            ],
            [
                "Vérifier l'installation du composant concerné.",
                "Réinstaller ou compléter sa configuration si nécessaire.",
                "Vérifier ensuite que le service fonctionne correctement.",
            ],
        ),

        (
            [
                "updated",
                "update",
                "upgraded",
            ],
            [
                "Vérifier la version actuelle du composant concerné.",
                "Appliquer la mise à jour appropriée puis vérifier son bon fonctionnement.",
            ],
        ),
    ]

    steps = None

    for keywords, candidate_steps in rules:

        if any(
            keyword in context_text
            for keyword in keywords
        ):
            steps = candidate_steps
            break

    if steps is None:

        steps = [
            "Analyser la configuration du composant concerné afin d'identifier la cause du problème.",
            "Appliquer la correction appropriée en fonction du diagnostic.",
            "Effectuer un test afin de confirmer la résolution de l'incident.",
        ]

    recommendation = (
        "Sur la base des incidents historiques similaires, "
        "il est recommandé de :\n\n"
    )

    recommendation += "\n".join(
        f"- {step}"
        for step in steps
    )

    return recommendation


# ============================================================
# 12. PIPELINE COMPLET
# ============================================================

@dataclass
class PipelineResult:

    status: str
    query: str

    recommendation: Optional[str] = None

    evidence: Optional[dict] = None

    winning_cluster: Optional[int] = None

    cluster_score: Optional[float] = None

    representative_ticket: Optional[str] = None

    representative_solution: Optional[str] = None

    candidates: Optional[pd.DataFrame] = None

    ranked_clusters: Optional[pd.DataFrame] = None

    message: Optional[str] = None

    # ----------------------------------------------------------
    # AJOUT -- indicateur de confiance basé sur la similarité
    # moyenne des tickets du CLUSTER GAGNANT (pas de l'ensemble
    # des candidats bruts -- un ticket très similaire mais hors
    # du cluster gagnant ne doit pas faire croire à une confiance
    # qu'il ne justifie pas). Champ additif : ne change ni le
    # status ni le comportement existant.
    # ----------------------------------------------------------

    max_ticket_similarity: Optional[float] = None

    confidence_level: Optional[str] = None  # "HIGH" | "MEDIUM" | "LOW"

    def to_dict(self) -> dict:
        return self.__dict__


# Seuils de confiance basés sur la distribution observée pendant le
# backtest (Ticket_Similarity moyenne ~0.76, médiane ~0.77). À
# ajuster si un backtest plus large donne une distribution différente.
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.65,
    "MEDIUM": 0.45,
}


def _confidence_level(max_similarity: Optional[float]) -> Optional[str]:
    """Classe la similarité max en HIGH / MEDIUM / LOW."""

    if max_similarity is None:
        return None

    if max_similarity >= CONFIDENCE_THRESHOLDS["HIGH"]:
        return "HIGH"

    if max_similarity >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"

    return "LOW"


def recommend_incident(
    query: str,
    knowledge_base: pd.DataFrame,
    ticket_embeddings: np.ndarray,
    solution_embeddings: Optional[np.ndarray] = None,
    model=None,
    top_k: int = 20,
    cluster_threshold: float = 0.75,
    exclude_ticket: Optional[str] = None,
    recommendation_generator=None,
) -> dict:
    """Pipeline complet de recommandation pour un ticket.

    Parameters
    ----------
    query:
        Texte du ticket à analyser.

    knowledge_base:
        Base historique complète.

    ticket_embeddings:
        Embeddings des tickets de knowledge_base.

    solution_embeddings:
        Embeddings pré-calculés des solutions.
        Utilisés principalement pendant le backtest.

    exclude_ticket:
        Ticket à exclure du retrieval.
        Obligatoire conceptuellement pour le leave-one-out
        du backtest.

    recommendation_generator:
        Fonction optionnelle (evidence: dict) -> str, appelée à
        l'étape 10 pour rédiger la recommandation finale.
        Par défaut : generate_natural_recommendation (heuristique).
        Pour utiliser le LLM local :
            from llm_local import generate_recommendation_llm
            recommend_incident(..., recommendation_generator=generate_recommendation_llm)
        Toute la logique de retrieval/filtrage/clustering/scoring
        (étapes 1 à 9) reste identique quel que soit le générateur
        utilisé -- seule la mise en forme du texte final change.

    Retour :
        dict avec status, query et résultats du pipeline.
    """
    if recommendation_generator is None:
        recommendation_generator = generate_natural_recommendation

    # --------------------------------------------------------
    # 1. RETRIEVAL
    # --------------------------------------------------------

    candidates = search_similar_tickets(
        query=query,
        knowledge_base=knowledge_base,
        ticket_embeddings=ticket_embeddings,
        model=model,
        top_k=top_k,
        exclude_ticket=exclude_ticket,
    )

    # --------------------------------------------------------
    # 2. FILTRAGE
    # --------------------------------------------------------

    actionable = filter_actionable_candidates(
        candidates
    )

    if actionable.empty:

        return PipelineResult(
            status="NO_ACTIONABLE_SOLUTION",
            query=query,
            candidates=candidates,
            message=(
                "Des tickets similaires existent mais "
                "aucune solution exploitable n'a été trouvée."
            ),
        ).to_dict()

    # --------------------------------------------------------
    # 3. SIMILARITÉ ENTRE SOLUTIONS
    #
    # En production :
    #     encodage à la volée.
    #
    # En backtest :
    #     utilisation des embeddings pré-calculés.
    # --------------------------------------------------------

    if solution_embeddings is not None:

        solution_similarity_matrix = (
            build_solution_similarity_matrix_from_precomputed(
                actionable,
                solution_embeddings
            )
        )

    else:

        solution_similarity_matrix = (
            build_solution_similarity_matrix_from_texts(
                actionable,
                model=model
            )
        )

    # --------------------------------------------------------
    # 4. CLUSTERING
    # --------------------------------------------------------

    clusters = cluster_similar_solutions(
        solution_similarity_matrix,
        threshold=cluster_threshold,
    )

    # --------------------------------------------------------
    # 5. SCORING
    # --------------------------------------------------------

    cluster_scores = score_solution_clusters(
        actionable,
        clusters,
        solution_similarity_matrix,
    )

    if cluster_scores.empty:

        return PipelineResult(
            status="NO_VALID_CLUSTER",
            query=query,
            candidates=actionable,
            message=(
                "Impossible de construire des clusters "
                "de solutions."
            ),
        ).to_dict()

    # --------------------------------------------------------
    # 6. RANKING
    # --------------------------------------------------------

    ranked_clusters = rank_solution_clusters(
        cluster_scores
    )

    # --------------------------------------------------------
    # 7. REPRESENTANTS
    # --------------------------------------------------------

    representatives = (
        select_cluster_representatives(
            actionable,
            clusters,
            solution_similarity_matrix,
        )
    )

    # --------------------------------------------------------
    # 8. MEILLEUR CLUSTER
    # --------------------------------------------------------

    best_result = select_best_cluster(
        ranked_clusters,
        representatives,
    )

    if best_result is None:

        return PipelineResult(
            status="NO_RECOMMENDATION",
            query=query,
            candidates=actionable,
            ranked_clusters=ranked_clusters,
            message=(
                "Impossible de sélectionner "
                "une recommandation."
            ),
        ).to_dict()

    # --------------------------------------------------------
    # 9. EVIDENCE
    # --------------------------------------------------------

    evidence = build_recommendation_evidence(
        query=query,
        best_result=best_result,
        clusters=clusters,
        actionable=actionable,
    )

    # --------------------------------------------------------
    # 10. RECOMMANDATION
    # --------------------------------------------------------

    recommendation = recommendation_generator(
        evidence
    )

    # --------------------------------------------------------
    # 11. INDICATEUR DE CONFIANCE
    #
    # IMPORTANT : basé sur les tickets du CLUSTER GAGNANT
    # (evidence), pas sur l'ensemble des candidats bruts.
    # Le max sur tous les candidats pouvait pointer vers un
    # ticket très similaire mais appartenant à un AUTRE cluster
    # que celui qui a produit la recommandation -- donnant une
    # fausse impression de confiance sur une recommandation
    # sans rapport avec ce ticket-là.
    #
    # On utilise la moyenne de similarité du cluster gagnant
    # (déjà calculée par build_recommendation_evidence), plus
    # robuste qu'un max isolé sur 1 seul ticket du cluster.
    # --------------------------------------------------------

    cluster_ticket_similarities = [
        ticket.get("Ticket_Similarity")
        for ticket in evidence.get("Evidence_Tickets", [])
        if isinstance(ticket.get("Ticket_Similarity"), (int, float))
    ]

    max_ticket_similarity = (
        max(cluster_ticket_similarities)
        if cluster_ticket_similarities
        else None
    )

    cluster_ticket_similarity_mean = evidence.get(
        "Ticket_Similarity_Mean"
    )

    # La confiance se base sur la moyenne du cluster (plus
    # robuste) quand elle est disponible, sinon sur le max.
    confidence_basis = (
        cluster_ticket_similarity_mean
        if cluster_ticket_similarity_mean is not None
        else max_ticket_similarity
    )

    confidence_level = _confidence_level(confidence_basis)

    return PipelineResult(
        status="SUCCESS",
        query=query,
        recommendation=recommendation,
        evidence=evidence,
        winning_cluster=best_result[
            "Cluster"
        ],
        cluster_score=best_result[
            "Cluster_Final_Score"
        ],
        representative_ticket=best_result[
            "Representative_Ticket"
        ],
        representative_solution=best_result[
            "Representative_Solution"
        ],
        candidates=actionable,
        ranked_clusters=ranked_clusters,
        max_ticket_similarity=confidence_basis,
        confidence_level=confidence_level,
    ).to_dict()


# ============================================================
# NOTE — OÙ BRANCHER LE LLM
# ============================================================
#
# generate_natural_recommendation() reste volontairement
# heuristique pour l'instant.
#
# Le LLM devra intervenir UNIQUEMENT après la sélection du
# cluster et du représentant.
#
# Le LLM recevra :
#
#   - le nouveau ticket
#   - Representative_Solution
#   - Evidence_Tickets
#   - scores
#
# et devra reformuler une recommandation concrète en se basant
# uniquement sur ces preuves.
#
# Le LLM ne doit PAS remplacer :
#
#   retrieval
#   filtering
#   clustering
#   scoring
#   representative selection
#
# Ces étapes restent déterministes et auditables.