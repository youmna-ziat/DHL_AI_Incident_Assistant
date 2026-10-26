"""Applique l'étape 2 sur src/recommendation_core.py :
  - recherche uniquement parmi les tickets exploitables (usable_only)
  - seuil de similarité minimal (statut LOW_SIMILARITY)
  - groupes habituels (similar_groups) en repli

Usage (depuis la racine du projet) :
    python patch_etape2.py
Une sauvegarde src/recommendation_core.py.bak_etape2 est créée.
"""
import shutil
import sys

P = "src/recommendation_core.py"

s = open(P, encoding="utf-8").read()

if "MIN_TICKET_SIMILARITY" in s:
    sys.exit("Déjà appliqué : rien à faire.")

shutil.copy(P, P + ".bak_etape2")


def rep(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(
            f"ERREUR : bloc introuvable ou ambigu ({s.count(old)} fois) :\n"
            f"{old[:90]!r}\nFichier non modifié (sauvegarde déjà faite)."
        )
    s = s.replace(old, new)


# 1) search_similar_tickets : option usable_only
rep('''    top_k: int = 20,
    exclude_ticket: Optional[str] = None,
) -> pd.DataFrame:
    """Retourne les top_k tickets historiques les plus proches.
''', '''    top_k: int = 20,
    exclude_ticket: Optional[str] = None,
    usable_only: bool = False,
) -> pd.DataFrame:
    """Retourne les top_k tickets historiques les plus proches.
''')

rep('''    exclude_ticket:
        Numéro du ticket à exclure.
        Indispensable pour le backtest leave-one-out.
    """

    query_embedding = compute_embeddings(''', '''    exclude_ticket:
        Numéro du ticket à exclure.
        Indispensable pour le backtest leave-one-out.

    usable_only:
        Si True, la recherche ne porte QUE sur les tickets dont la
        solution est exploitable (ACTIONABLE et informative). Le
        top-k est alors pris parmi ces tickets, au lieu de filtrer
        après coup (ce qui laissait très peu de candidats).
    """

    query_embedding = compute_embeddings(''')

rep('''    valid_indices = np.flatnonzero(valid_mask)

    if len(valid_indices) == 0:''', '''    if usable_only and "solution_type" in knowledge_base.columns:
        usable = (
            knowledge_base["solution_type"] == "ACTIONABLE"
        ).to_numpy()

        if "non_informative" in knowledge_base.columns:
            usable = usable & ~(
                knowledge_base["non_informative"]
                .astype(bool)
                .to_numpy()
            )

        valid_mask = valid_mask & usable

    valid_indices = np.flatnonzero(valid_mask)

    if len(valid_indices) == 0:''')

# 2) helpers groupes
rep('''# ============================================================
# 5. FILTRAGE DES SOLUTIONS EXPLOITABLES''', '''def get_similar_groups(
    query: str,
    knowledge_base: pd.DataFrame,
    ticket_embeddings: np.ndarray,
    model=None,
    top_k: int = 15,
    n_groups: int = 3,
    exclude_ticket: Optional[str] = None,
) -> list:
    """Groupes qui ont traité les incidents les plus proches.

    La recherche porte sur TOUS les tickets (même ceux dont la
    solution est vide ou générique) : savoir QUI traite ce type
    d'incident reste utile quand aucune solution n'est documentée.
    """

    neighbours = search_similar_tickets(
        query=query,
        knowledge_base=knowledge_base,
        ticket_embeddings=ticket_embeddings,
        model=model,
        top_k=top_k,
        exclude_ticket=exclude_ticket,
        usable_only=False,
    )

    if neighbours.empty or "Assigned Group" not in neighbours.columns:
        return []

    counts = (
        neighbours["Assigned Group"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(n_groups)
    )

    return [
        {"Group": group, "Tickets": int(n)}
        for group, n in counts.items()
    ]


def _groups_sentence(groups: Optional[list]) -> str:
    """Phrase lisible sur les groupes habituels (peut être vide)."""

    if not groups:
        return ""

    names = ", ".join(
        f"{g['Group']} ({g['Tickets']} tickets proches)"
        for g in groups
    )

    return (
        " Les incidents les plus proches ont surtout été traités par : "
        f"{names}."
    )


# ============================================================
# 5. FILTRAGE DES SOLUTIONS EXPLOITABLES''')

# 3) PipelineResult : nouveau champ
rep('''    confidence_level: Optional[str] = None  # "HIGH" | "MEDIUM" | "LOW"
''', '''    confidence_level: Optional[str] = None  # "HIGH" | "MEDIUM" | "LOW"

    # Groupes ayant traité les incidents les plus proches (additif).
    similar_groups: Optional[list] = None
''')

# 4) constante de seuil
rep('''def _confidence_level(max_similarity''', '''# En dessous de cette similarité (meilleur ticket candidat), le
# pipeline REFUSE de recommander : l'historique ne contient rien de
# comparable. Valeur de départ à calibrer avec le backtest.
MIN_TICKET_SIMILARITY = 0.40


def _confidence_level(max_similarity''')

# 5) signature de recommend_incident
rep('''    exclude_ticket: Optional[str] = None,
    recommendation_generator=None,
) -> dict:''', '''    exclude_ticket: Optional[str] = None,
    recommendation_generator=None,
    usable_only: bool = True,
    min_ticket_similarity: Optional[float] = MIN_TICKET_SIMILARITY,
) -> dict:''')

# 6) retrieval + garde-fou + filtrage
rep('''    candidates = search_similar_tickets(
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
''', '''    candidates = search_similar_tickets(
        query=query,
        knowledge_base=knowledge_base,
        ticket_embeddings=ticket_embeddings,
        model=model,
        top_k=top_k,
        exclude_ticket=exclude_ticket,
        usable_only=usable_only,
    )

    # Groupes habituels (repli utile même sans solution documentée).
    similar_groups = get_similar_groups(
        query=query,
        knowledge_base=knowledge_base,
        ticket_embeddings=ticket_embeddings,
        model=model,
        exclude_ticket=exclude_ticket,
    )

    if candidates.empty:

        return PipelineResult(
            status="NO_ACTIONABLE_SOLUTION",
            query=query,
            candidates=candidates,
            message=(
                "Aucune solution exploitable n'existe dans l'historique."
                + _groups_sentence(similar_groups)
            ),
            similar_groups=similar_groups,
        ).to_dict()

    # --------------------------------------------------------
    # 1bis. GARDE-FOU : similarité minimale
    #
    # Si même le meilleur ticket est trop éloigné, mieux vaut dire
    # "je ne sais pas" que proposer une recommandation sans rapport.
    # --------------------------------------------------------

    best_similarity = float(candidates["Ticket_Similarity"].max())

    if (
        min_ticket_similarity is not None
        and best_similarity < min_ticket_similarity
    ):

        return PipelineResult(
            status="LOW_SIMILARITY",
            query=query,
            candidates=candidates,
            max_ticket_similarity=best_similarity,
            confidence_level="LOW",
            message=(
                "Aucun incident vraiment comparable n'a été trouvé "
                f"dans l'historique (meilleure similarité : "
                f"{best_similarity:.2f}). Aucune recommandation "
                "n'est proposée pour éviter une réponse hors sujet."
                + _groups_sentence(similar_groups)
            ),
            similar_groups=similar_groups,
        ).to_dict()

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
                + _groups_sentence(similar_groups)
            ),
            similar_groups=similar_groups,
        ).to_dict()
''')

# 7) retour SUCCESS : ajouter similar_groups
rep('''        max_ticket_similarity=confidence_basis,
        confidence_level=confidence_level,
    ).to_dict()''', '''        max_ticket_similarity=confidence_basis,
        confidence_level=confidence_level,
        similar_groups=similar_groups,
    ).to_dict()''')

open(P, "w", encoding="utf-8").write(s)
print("OK : étape 2 appliquée. Sauvegarde :", P + ".bak_etape2")
