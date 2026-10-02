"""
03_backtest.py
==============

BACKTEST LEAVE-ONE-OUT DU PIPELINE DE RECOMMANDATION DHL.

PRINCIPE
--------
Pour chaque ticket testé :

1. Le ticket est utilisé comme requête.
2. Il est exclu du retrieval (leave-one-out).
3. Le pipeline de PRODUCTION recommend_incident() est exécuté.
4. La solution recommandée est comparée à la vraie solution historique.
5. Une baseline top-1 naïve est également calculée.

IMPORTANT
---------
Ce script ne réimplémente AUCUNE logique du pipeline.

Toute la logique métier reste centralisée dans :

    src/recommendation_core.py

Le backtest appelle uniquement :

    recommendation_core.recommend_incident()

Cela garantit que le pipeline testé est exactement le pipeline
utilisé en production.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# IMPORT DU MODULE CENTRAL
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import recommendation_core as rc  # noqa: E402


# ================================================================
# CONFIGURATION
# ================================================================

DATA_DIR = PROJECT_ROOT / "data"

RESOLVED_PATH = (
    DATA_DIR / "Inc_resolved by MAIT.xlsx"
)

ESCALATED_PATH = (
    DATA_DIR / "Inc_registred by MAIT.xlsx"
)

TICKET_EMBEDDINGS_PATH = (
    DATA_DIR / "ticket_embeddings.npy"
)

SOLUTION_EMBEDDINGS_PATH = (
    DATA_DIR / "solution_embeddings.npy"
)


TOP_K = 20

CLUSTER_THRESHOLD = 0.75

N_TEST_TICKETS = 200

RANDOM_SEED = 42


# ================================================================
# VALIDATION DES FICHIERS
# ================================================================

def validate_input_files() -> None:
    """Vérifie que les fichiers Excel nécessaires existent."""

    missing_files = []

    if not RESOLVED_PATH.exists():
        missing_files.append(str(RESOLVED_PATH))

    if not ESCALATED_PATH.exists():
        missing_files.append(str(ESCALATED_PATH))

    if missing_files:

        raise FileNotFoundError(
            "Fichier(s) introuvable(s) :\n"
            + "\n".join(
                f" - {path}"
                for path in missing_files
            )
        )


# ================================================================
# 1. CHARGEMENT / CRÉATION DES EMBEDDINGS
# ================================================================

def load_or_build_embeddings(
    knowledge_base: pd.DataFrame,
    model,
):
    """
    Réutilise les embeddings pré-calculés lorsqu'ils sont compatibles
    avec la knowledge base actuelle.

    Sinon :

    - recalcule les embeddings des tickets
    - recalcule les embeddings des solutions
    - sauvegarde les nouveaux fichiers .npy

    IMPORTANT :
    Les embeddings doivent être parfaitement alignés avec KB_Index.
    """

    need_rebuild = True

    ticket_embeddings = None
    solution_embeddings = None


    # ------------------------------------------------------------
    # Tentative de chargement
    # ------------------------------------------------------------

    if (
        TICKET_EMBEDDINGS_PATH.exists()
        and SOLUTION_EMBEDDINGS_PATH.exists()
    ):

        try:

            ticket_embeddings = np.load(
                TICKET_EMBEDDINGS_PATH
            )

            solution_embeddings = np.load(
                SOLUTION_EMBEDDINGS_PATH
            )


            # ----------------------------------------------------
            # Vérification minimale d'alignement
            # ----------------------------------------------------

            valid_ticket_shape = (
                ticket_embeddings.ndim == 2
                and ticket_embeddings.shape[0]
                == len(knowledge_base)
            )

            valid_solution_shape = (
                solution_embeddings.ndim == 2
                and solution_embeddings.shape[0]
                == len(knowledge_base)
            )


            if (
                valid_ticket_shape
                and valid_solution_shape
            ):

                print(
                    "Embeddings existants réutilisés."
                )

                print(
                    f"Ticket embeddings : "
                    f"{ticket_embeddings.shape}"
                )

                print(
                    f"Solution embeddings : "
                    f"{solution_embeddings.shape}"
                )

                need_rebuild = False


        except Exception as error:

            print(
                "Impossible de charger les embeddings existants."
            )

            print(
                f"Raison : {error}"
            )

            need_rebuild = True


    # ------------------------------------------------------------
    # Recalcul
    # ------------------------------------------------------------

    if need_rebuild:

        print(
            "\nEmbeddings absents ou incompatibles."
        )

        print(
            "Recalcul des embeddings..."
        )


        ticket_texts = (
            knowledge_base[
                "text_for_embedding"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )


        solution_texts = (
            knowledge_base[
                "Solution"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )


        ticket_embeddings = (
            rc.compute_embeddings(
                ticket_texts,
                model=model,
            )
        )


        solution_embeddings = (
            rc.compute_embeddings(
                solution_texts,
                model=model,
            )
        )


        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


        np.save(
            TICKET_EMBEDDINGS_PATH,
            ticket_embeddings,
        )


        np.save(
            SOLUTION_EMBEDDINGS_PATH,
            solution_embeddings,
        )


        print(
            "Embeddings recalculés et sauvegardés."
        )

        print(
            f"Ticket embeddings : "
            f"{ticket_embeddings.shape}"
        )

        print(
            f"Solution embeddings : "
            f"{solution_embeddings.shape}"
        )


    return (
        ticket_embeddings,
        solution_embeddings,
    )


# ================================================================
# 2. OUTILS DE SIMILARITÉ
# ================================================================

def cosine_sim(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )


    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )


    if denominator == 0:

        return 0.0


    return float(
        np.dot(a, b)
        / denominator
    )


# ================================================================
# 3. BACKTEST D'UN SEUL TICKET
# ================================================================

def run_single_backtest(

    ticket_number: str,

    knowledge_base: pd.DataFrame,

    ticket_embeddings: np.ndarray,

    solution_embeddings: np.ndarray,

    model,

    top_k: int = TOP_K,

    cluster_threshold: float = CLUSTER_THRESHOLD,

) -> dict:
    """
    Teste le pipeline de PRODUCTION sur un ticket historique.

    Leave-one-out :

        le ticket testé est exclu du retrieval.

    IMPORTANT :

        Cette fonction ne contient aucune logique de recommandation.

        Elle appelle uniquement :

            rc.recommend_incident()

        puis mesure la qualité du résultat.
    """


    # ------------------------------------------------------------
    # Recherche du ticket testé
    # ------------------------------------------------------------

    test_rows = knowledge_base.loc[

        knowledge_base["Number"]
        .astype(str)
        == str(ticket_number)

    ]


    if test_rows.empty:

        return {

            "Test_Ticket": ticket_number,

            "Status": "TICKET_NOT_FOUND",

        }


    # ------------------------------------------------------------
    # Vérification d'unicité
    # ------------------------------------------------------------

    if len(test_rows) > 1:

        return {

            "Test_Ticket": ticket_number,

            "Status": "DUPLICATE_TICKET_NUMBER",

        }


    test_row = test_rows.iloc[0]


    # ------------------------------------------------------------
    # Vérité terrain
    # ------------------------------------------------------------

    test_kb_index = int(
        test_row["KB_Index"]
    )


    true_solution = (

        ""

        if pd.isna(
            test_row["Solution"]
        )

        else str(
            test_row["Solution"]
        )

    )


    true_solution_embedding = (

        solution_embeddings[
            test_kb_index
        ]

    )


    query = str(
        test_row["text_for_embedding"]
    )


    # ============================================================
    # PIPELINE DE PRODUCTION
    # ============================================================

    result = rc.recommend_incident(

        query=query,

        knowledge_base=knowledge_base,

        ticket_embeddings=ticket_embeddings,

        solution_embeddings=solution_embeddings,

        model=model,

        top_k=top_k,

        cluster_threshold=cluster_threshold,

        exclude_ticket=ticket_number,

    )


    # ------------------------------------------------------------
    # Enregistrement de base
    # ------------------------------------------------------------

    record = {

        "Test_Ticket":
            ticket_number,

        "Status":
            result["status"],

        "True_Solution":
            true_solution,

    }


    # ------------------------------------------------------------
    # Le pipeline n'a pas produit de recommandation
    # ------------------------------------------------------------

    if result["status"] != "SUCCESS":

        record.update({

            "Solution_Similarity":
                None,

            "Ticket_Similarity":
                None,

            "Representative_Ticket":
                None,

            "Representative_Solution":
                None,

            "Cluster_Score":
                None,

            "Number_of_Tickets_in_Cluster":
                None,

            "Baseline_Top1_Solution_Similarity":
                None,

            "Baseline_Top1_Ticket_Similarity":
                None,

        })


        return record


    # ============================================================
    # ÉVALUATION DE LA RECOMMANDATION
    # ============================================================

    candidates = result["candidates"]

    representative_ticket = (
        result["representative_ticket"]
    )


    repr_rows = candidates.loc[

        candidates["Number"]
        .astype(str)
        == str(representative_ticket)

    ]


    if repr_rows.empty:

        record[
            "Solution_Similarity"
        ] = None

        record[
            "Ticket_Similarity"
        ] = None


    else:

        repr_row = (
            repr_rows.iloc[0]
        )


        repr_kb_index = int(
            repr_row["KB_Index"]
        )


        repr_solution_embedding = (

            solution_embeddings[
                repr_kb_index
            ]

        )


        record[
            "Solution_Similarity"
        ] = cosine_sim(

            true_solution_embedding,

            repr_solution_embedding,

        )


        record[
            "Ticket_Similarity"
        ] = float(

            repr_row[
                "Ticket_Similarity"
            ]

        )


    record[
        "Representative_Ticket"
    ] = representative_ticket


    record[
        "Representative_Solution"
    ] = result[
        "representative_solution"
    ]


    record[
        "Cluster_Score"
    ] = result[
        "cluster_score"
    ]


    evidence = (
        result.get("evidence")
    )


    if evidence is not None:

        record[
            "Number_of_Tickets_in_Cluster"
        ] = evidence.get(
            "Number_of_Tickets"
        )

    else:

        record[
            "Number_of_Tickets_in_Cluster"
        ] = None


    # ============================================================
    # BASELINE TOP-1 NAÏVE
    # ============================================================

    naive_candidates = (

        rc.search_similar_tickets(

            query=query,

            knowledge_base=knowledge_base,

            ticket_embeddings=ticket_embeddings,

            model=model,

            top_k=1,

            exclude_ticket=ticket_number,

        )

    )


    if naive_candidates.empty:

        record[
            "Baseline_Top1_Solution_Similarity"
        ] = None


        record[
            "Baseline_Top1_Ticket_Similarity"
        ] = None


    else:

        naive_row = (
            naive_candidates.iloc[0]
        )


        naive_kb_index = int(
            naive_row["KB_Index"]
        )


        naive_solution_embedding = (

            solution_embeddings[
                naive_kb_index
            ]

        )


        record[
            "Baseline_Top1_Solution_Similarity"
        ] = cosine_sim(

            true_solution_embedding,

            naive_solution_embedding,

        )


        record[
            "Baseline_Top1_Ticket_Similarity"
        ] = float(

            naive_row[
                "Ticket_Similarity"
            ]

        )


    return record


# ================================================================
# 4. SÉLECTION DES TICKETS À TESTER
# ================================================================

def sample_test_tickets(

    knowledge_base: pd.DataFrame,

    n: int,

    seed: int = RANDOM_SEED,

) -> list[str]:
    """
    Sélectionne les tickets utilisés pour le backtest.

    On teste uniquement les tickets :

        - ACTIONABLE
        - informatifs

    Pourquoi ?

    Une solution comme :

        "Resolved"

    ne constitue pas une vérité terrain exploitable.

    La question évaluée est donc :

        "Lorsqu'une vraie solution technique historique existe,
         le pipeline retrouve-t-il une solution similaire ?"
    """


    pool = knowledge_base.loc[

        (
            knowledge_base[
                "solution_type"
            ]
            == "ACTIONABLE"
        )

        &

        (
            ~knowledge_base[
                "non_informative"
            ]
        )

    ].copy()


    if pool.empty:

        raise ValueError(

            "Aucun ticket ACTIONABLE et informatif "
            "n'est disponible pour le backtest."

        )


    n = min(
        n,
        len(pool),
    )


    if n <= 0:

        raise ValueError(

            "Le nombre de tickets à tester doit être supérieur à 0."

        )


    sample = (

        pool.sample(

            n=n,

            random_state=seed,

        )

    )


    return (

        sample[
            "Number"
        ]
        .astype(str)
        .tolist()

    )


# ================================================================
# 5. BACKTEST COMPLET
# ================================================================

def run_backtest(

    knowledge_base: pd.DataFrame,

    ticket_embeddings: np.ndarray,

    solution_embeddings: np.ndarray,

    model,

    n_tickets: int = N_TEST_TICKETS,

) -> pd.DataFrame:
    """
    Exécute le backtest sur un échantillon de tickets.
    """


    test_tickets = (

        sample_test_tickets(

            knowledge_base,

            n_tickets,

        )

    )


    print()

    print(
        f"Backtest sur "
        f"{len(test_tickets)} tickets."
    )

    print(
        "Méthode : Leave-One-Out"
    )

    print()


    records = []


    for i, ticket in enumerate(

        test_tickets,

        start=1,

    ):

        record = (

            run_single_backtest(

                ticket_number=ticket,

                knowledge_base=knowledge_base,

                ticket_embeddings=ticket_embeddings,

                solution_embeddings=solution_embeddings,

                model=model,

            )

        )


        records.append(
            record
        )


        if (

            i % 10 == 0

            or i == len(
                test_tickets
            )

        ):

            print(

                f"  ... "
                f"{i}/{len(test_tickets)} "
                f"tickets testés"

            )


    return pd.DataFrame(
        records
    )


# ================================================================
# 6. RÉSUMÉ DES RÉSULTATS
# ================================================================

def print_summary(
    results: pd.DataFrame,
) -> None:
    """
    Affiche les métriques principales du backtest.
    """


    print()

    print("=" * 70)

    print(
        "RÉSUMÉ DU BACKTEST"
    )

    print("=" * 70)


    # ------------------------------------------------------------
    # Répartition des statuts
    # ------------------------------------------------------------

    print()

    print(
        "Répartition des status :"
    )

    print(
        results[
            "Status"
        ]
        .value_counts()
    )


    # ------------------------------------------------------------
    # Succès
    # ------------------------------------------------------------

    success = (

        results.loc[

            results[
                "Status"
            ]
            == "SUCCESS"

        ]

        .copy()

    )


    n_success = len(
        success
    )

    n_total = len(
        results
    )


    if n_total == 0:

        print(
            "\nAucun ticket testé."
        )

        return


    success_rate = (

        100
        * n_success
        / n_total

    )


    print()

    print(

        "Taux de succès "
        "(pipeline a produit une recommandation) : "

        f"{n_success}/{n_total} "

        f"({success_rate:.1f}%)"

    )


    if n_success == 0:

        print()

        print(
            "Aucun succès : impossible de calculer "
            "les métriques de qualité."
        )

        return


    # ------------------------------------------------------------
    # Qualité du pipeline
    # ------------------------------------------------------------

    valid_solution_scores = (

        success[
            "Solution_Similarity"
        ]
        .dropna()

    )


    valid_ticket_scores = (

        success[
            "Ticket_Similarity"
        ]
        .dropna()

    )


    print()

    print(
        "--- QUALITÉ DU PIPELINE AVEC CLUSTERING ---"
    )


    if not valid_solution_scores.empty:

        print(

            "Solution_Similarity : "

            f"moyenne="
            f"{valid_solution_scores.mean():.3f} "

            f"| médiane="
            f"{valid_solution_scores.median():.3f}"

        )


    if not valid_ticket_scores.empty:

        print(

            "Ticket_Similarity : "

            f"moyenne="
            f"{valid_ticket_scores.mean():.3f} "

            f"| médiane="
            f"{valid_ticket_scores.median():.3f}"

        )


    # ------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------

    baseline = (

        success[
            "Baseline_Top1_Solution_Similarity"
        ]
        .dropna()

    )


    print()

    print(
        "--- BASELINE TOP-1 NAÏVE ---"
    )


    if not baseline.empty:

        print(

            "Solution_Similarity : "

            f"moyenne="
            f"{baseline.mean():.3f} "

            f"| médiane="
            f"{baseline.median():.3f}"

        )


        # Comparaison uniquement sur les tickets
        # ayant les deux métriques disponibles.

        comparison = (

            success[

                [

                    "Solution_Similarity",

                    "Baseline_Top1_Solution_Similarity",

                ]

            ]

            .dropna()

        )


        if not comparison.empty:

            gain = (

                comparison[
                    "Solution_Similarity"
                ].mean()

                -

                comparison[
                    "Baseline_Top1_Solution_Similarity"
                ].mean()

            )


            print()

            print(

                "Gain moyen du pipeline "

                f"vs baseline : {gain:+.3f}"

            )


            if gain > 0.02:

                print(

                    "-> Le pipeline avec clustering "
                    "apporte un gain de qualité "
                    "sur cet échantillon."

                )


            elif gain < -0.02:

                print(

                    "-> ATTENTION : le pipeline fait moins bien "
                    "que la baseline top-1 sur cet échantillon."

                )

                print(

                    "   À analyser : TOP_K, "
                    "CLUSTER_THRESHOLD, "
                    "qualité des solutions historiques."

                )


            else:

                print(

                    "-> Performance similaire à la baseline."

                )

                print(

                    "   La valeur ajoutée du clustering peut alors "
                    "être principalement la robustesse et "
                    "l'explicabilité."

                )


    # ------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------

    print()

    print(
        "--- DISTRIBUTION DE LA SOLUTION_SIMILARITY ---"
    )


    if not valid_solution_scores.empty:

        bins = [

            -1.01,

            0.30,

            0.50,

            0.70,

            1.01,

        ]


        labels = [

            "< 0.30 (faible)",

            "0.30 - 0.50",

            "0.50 - 0.70",

            ">= 0.70 (bon)",

        ]


        distribution = (

            pd.cut(

                valid_solution_scores,

                bins=bins,

                labels=labels,

            )

            .value_counts()

            .sort_index()

        )


        print(
            distribution
        )


# ================================================================
# 7. POINT D'ENTRÉE
# ================================================================

def main() -> None:

    print()

    print("=" * 70)

    print(
        "DHL AI INCIDENT ASSISTANT"
    )

    print(
        "BACKTEST LEAVE-ONE-OUT"
    )

    print("=" * 70)


    # ------------------------------------------------------------
    # Vérification des fichiers
    # ------------------------------------------------------------

    validate_input_files()


    # ------------------------------------------------------------
    # Chargement de la KB
    # ------------------------------------------------------------

    print()

    print(
        "Chargement de la base de connaissance..."
    )


    knowledge_base = (

        rc.load_knowledge_base(

            str(
                RESOLVED_PATH
            ),

            str(
                ESCALATED_PATH
            ),

        )

    )


    knowledge_base = (

        rc.add_solution_types(
            knowledge_base
        )

    )


    print()

    print(
        f"Base chargée : "
        f"{knowledge_base.shape}"
    )


    print()

    print(
        "Répartition des types de solutions :"
    )


    print(

        knowledge_base[
            "solution_type"
        ]
        .value_counts()

    )


    # ------------------------------------------------------------
    # Vérification des numéros dupliqués
    # ------------------------------------------------------------

    duplicate_numbers = (

        knowledge_base[
            "Number"
        ]

        .duplicated()

        .sum()

    )


    if duplicate_numbers > 0:

        print()

        print(
            "ATTENTION : "
            f"{duplicate_numbers} numéro(s) de ticket dupliqué(s)."
        )

        print(
            "Le leave-one-out par Number peut exclure "
            "plusieurs lignes."
        )


    # ------------------------------------------------------------
    # Chargement du modèle
    # ------------------------------------------------------------

    print()

    print(
        "Chargement du modèle d'embeddings..."
    )


    model = (
        rc.get_model()
    )


    # ------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------

    ticket_embeddings, solution_embeddings = (

        load_or_build_embeddings(

            knowledge_base,

            model,

        )

    )


    # ------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------

    results = (

        run_backtest(

            knowledge_base,

            ticket_embeddings,

            solution_embeddings,

            model,

        )

    )


    # ------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------

    DATA_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    output_path = (

        DATA_DIR
        / "backtest_results.csv"

    )


    results.to_csv(

        output_path,

        index=False,

    )


    print()

    print(

        "Résultats détaillés sauvegardés dans :"

    )

    print(
        output_path
    )


    # ------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------

    print_summary(
        results
    )


if __name__ == "__main__":

    main()