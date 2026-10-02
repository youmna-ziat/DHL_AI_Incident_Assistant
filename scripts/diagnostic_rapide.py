"""
diagnostic_rapide.py
====================

Analyse le fichier data/backtest_results.csv déjà produit par
03_backtest.py.

Ne relance PAS le backtest.
Ne recalcule PAS les résultats du pipeline.

Objectif : comprendre pourquoi le pipeline avec clustering obtient
une performance similaire à la baseline Top-1 naïve.

Questions analysées :

1. Les clusters gagnants sont-ils principalement des singletons
   (un seul ticket) ?

2. Existe-t-il une relation entre la similarité du ticket et la
   similarité de la solution ?

3. Quel est le niveau de similarité entre des solutions choisies
   aléatoirement ?

Ce script sert uniquement au diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION DES CHEMINS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "backtest_results.csv"
SRC_DIR = PROJECT_ROOT / "src"

RESOLVED_PATH = PROJECT_ROOT / "data" / "Inc_resolved by MAIT.xlsx"
ESCALATED_PATH = PROJECT_ROOT / "data" / "Inc_registred by MAIT.xlsx"

SOLUTION_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "solution_embeddings.npy"
)


# ================================================================
# CHARGEMENT DES RÉSULTATS
# ================================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {CSV_PATH}\n"
        "Exécutez d'abord : python scripts/03_backtest.py"
    )

df = pd.read_csv(CSV_PATH)

if df.empty:
    raise ValueError(
        "Le fichier backtest_results.csv est vide."
    )

if "Status" not in df.columns:
    raise ValueError(
        "La colonne 'Status' est absente du fichier de backtest."
    )

success = df[df["Status"] == "SUCCESS"].copy()

if success.empty:
    print("Aucun résultat SUCCESS dans le backtest.")
    sys.exit(0)


print("\n" + "=" * 70)
print("DIAGNOSTIC RAPIDE DU BACKTEST")
print("=" * 70)

print(f"\nFichier analysé : {CSV_PATH}")
print(f"Nombre total de tickets : {len(df)}")
print(f"Nombre de succès        : {len(success)}")


# ================================================================
# 1. TAILLE DES CLUSTERS GAGNANTS
# ================================================================

print("\n" + "=" * 70)
print("1. TAILLE DES CLUSTERS GAGNANTS")
print("=" * 70)

required_column = "Number_of_Tickets_in_Cluster"

if required_column not in success.columns:

    print(
        f"\nColonne absente : {required_column}"
    )
    print(
        "Impossible d'analyser la taille des clusters."
    )

else:

    cluster_sizes = (
        success[required_column]
        .value_counts()
        .sort_index()
    )

    print("\nDistribution :")
    print(cluster_sizes)

    singleton_pct = (
        success[required_column] == 1
    ).mean() * 100

    print(
        f"\n% de clusters gagnants avec 1 seul ticket : "
        f"{singleton_pct:.1f}%"
    )

    multi_ticket_pct = (
        success[required_column] >= 2
    ).mean() * 100

    print(
        f"% de clusters avec au moins 2 tickets       : "
        f"{multi_ticket_pct:.1f}%"
    )

    if singleton_pct >= 80:

        print(
            "\nINTERPRÉTATION :"
        )

        print(
            "La majorité des clusters gagnants sont des singletons."
        )

        print(
            "Dans cette configuration, le clustering a très peu"
        )

        print(
            "d'opportunités pour améliorer le résultat par rapport"
        )

        print(
            "à un simple Top-1."
        )

        print(
            "Ce n'est pas nécessairement un bug."
        )

    elif singleton_pct >= 50:

        print(
            "\nINTERPRÉTATION :"
        )

        print(
            "Les singletons sont fréquents."
        )

        print(
            "Le clustering peut aider dans certains cas,"
        )

        print(
            "mais son impact moyen risque d'être limité."
        )

    else:

        print(
            "\nINTERPRÉTATION :"
        )

        print(
            "Les clusters contiennent souvent plusieurs tickets."
        )

        print(
            "Le clustering a donc réellement la possibilité"
        )

        print(
            "d'apporter une valeur ajoutée."
        )


# ================================================================
# 2. RELATION TICKET_SIMILARITY / SOLUTION_SIMILARITY
# ================================================================

print("\n" + "=" * 70)
print("2. RELATION ENTRE TICKET_SIMILARITY ET SOLUTION_SIMILARITY")
print("=" * 70)

required_columns = [
    "Ticket_Similarity",
    "Solution_Similarity",
]

missing_columns = [
    column
    for column in required_columns
    if column not in success.columns
]

if missing_columns:

    print(
        f"\nColonnes manquantes : {missing_columns}"
    )

else:

    analysis = success[
        required_columns
    ].dropna()

    if len(analysis) < 2:

        print(
            "\nPas assez de données pour calculer une corrélation."
        )

    else:

        corr = analysis[
            "Ticket_Similarity"
        ].corr(
            analysis["Solution_Similarity"]
        )

        print(
            f"\nCorrélation Pearson : {corr:.3f}"
        )

        print(
            f"Nombre de cas analysés : {len(analysis)}"
        )

        if corr >= 0.50:

            print(
                "\nINTERPRÉTATION :"
            )

            print(
                "Une forte similarité du ticket est généralement"
            )

            print(
                "associée à une solution historiquement similaire."
            )

        elif corr >= 0.20:

            print(
                "\nINTERPRÉTATION :"
            )

            print(
                "Il existe une relation positive, mais modérée,"
            )

            print(
                "entre la similarité du ticket et celle de la solution."
            )

        elif corr > -0.20:

            print(
                "\nINTERPRÉTATION :"
            )

            print(
                "La relation est faible."
            )

            print(
                "Trouver un ticket similaire ne garantit pas"
            )

            print(
                "nécessairement de retrouver une solution similaire."
            )

        else:

            print(
                "\nINTERPRÉTATION :"
            )

            print(
                "La relation est faible ou négative."
            )

            print(
                "Cela suggère une forte variabilité dans les"
            )

            print(
                "textes de solutions."
            )

        print(
            "\nTop 10 cas avec Ticket_Similarity la plus haute :"
        )

        display_columns = [
            "Test_Ticket",
            "Ticket_Similarity",
            "Solution_Similarity",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in success.columns
        ]

        print(
            success.nlargest(
                10,
                "Ticket_Similarity"
            )[available_columns].to_string(
                index=False
            )
        )


# ================================================================
# 3. BASELINE ALÉATOIRE
# ================================================================

print("\n" + "=" * 70)
print("3. BASELINE ALÉATOIRE")
print("=" * 70)

try:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )

    import recommendation_core as rc

    if not SOLUTION_EMBEDDINGS_PATH.exists():

        raise FileNotFoundError(
            f"Embeddings introuvables : "
            f"{SOLUTION_EMBEDDINGS_PATH}"
        )

    print("\nChargement de la base de connaissance...")

    kb = rc.load_knowledge_base(
        str(RESOLVED_PATH),
        str(ESCALATED_PATH),
    )

    kb = rc.add_solution_types(kb)

    print(
        "Chargement des solution embeddings..."
    )

    solution_embeddings = np.load(
        SOLUTION_EMBEDDINGS_PATH
    )

    if len(solution_embeddings) != len(kb):

        raise ValueError(
            "Nombre d'embeddings incompatible avec "
            "la base de connaissance."
        )

    rng = np.random.default_rng(42)

    tested_numbers = (
        success["Test_Ticket"]
        .astype(str)
        .tolist()
    )

    tested_rows = kb[
        kb["Number"]
        .astype(str)
        .isin(tested_numbers)
    ]

    tested_kb_indices = (
        tested_rows["KB_Index"]
        .astype(int)
        .to_numpy()
    )

    random_scores = []

    all_indices = np.arange(
        len(kb)
    )

    for idx in tested_kb_indices:

        other = rng.choice(
            all_indices
        )

        while other == idx:

            other = rng.choice(
                all_indices
            )

        a = solution_embeddings[idx]
        b = solution_embeddings[other]

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:

            score = 0.0

        else:

            score = float(
                np.dot(a, b)
                / denominator
            )

        random_scores.append(score)

    random_scores = np.asarray(
        random_scores
    )

    pipeline_scores = (
        success[
            "Solution_Similarity"
        ]
        .dropna()
    )

    print(
        "\nRÉSULTATS :"
    )

    print(
        f"Similarité moyenne aléatoire : "
        f"{random_scores.mean():.3f}"
    )

    print(
        f"Similarité médiane aléatoire : "
        f"{np.median(random_scores):.3f}"
    )

    print(
        f"Similarité moyenne pipeline  : "
        f"{pipeline_scores.mean():.3f}"
    )

    print(
        f"Similarité médiane pipeline  : "
        f"{pipeline_scores.median():.3f}"
    )

    gain_random = (
        pipeline_scores.mean()
        - random_scores.mean()
    )

    print(
        f"\nGain pipeline vs hasard : "
        f"{gain_random:+.3f}"
    )

    if gain_random >= 0.15:

        print(
            "\nINTERPRÉTATION :"
        )

        print(
            "Le pipeline fait significativement mieux que le hasard."
        )

    elif gain_random >= 0.05:

        print(
            "\nINTERPRÉTATION :"
        )

        print(
            "Le pipeline fait clairement mieux que le hasard,"
        )

        print(
            "mais il existe encore une marge d'amélioration."
        )

    else:

        print(
            "\nINTERPRÉTATION :"
        )

        print(
            "Le pipeline est proche du niveau aléatoire."
        )

        print(
            "Il faudrait alors examiner le retrieval,"
        )

        print(
            "la classification des solutions ou la métrique."
        )

except Exception as e:

    print(
        f"\nÉtape 3 impossible : {e}"
    )


# ================================================================
# FIN
# ================================================================

print("\n" + "=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)