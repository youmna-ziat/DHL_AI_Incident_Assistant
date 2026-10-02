"""
04_llm_demo.py
===============

Compare la recommandation HEURISTIQUE (par mots-clés, celle utilisée
par défaut dans recommend_incident) et la recommandation LLM LOCAL
(Qwen2.5-1.5B-Instruct) sur 2-3 exemples, pour valider visuellement
que le LLM apporte un vrai gain de qualité avant de le brancher au
frontend.

LANCER :
    python scripts/04_llm_demo.py

PREMIER LANCEMENT : le modèle LLM (~3 Go) va se télécharger, ça peut
prendre quelques minutes selon la connexion. Les lancements suivants
sont rapides (modèle mis en cache par HuggingFace).
"""

from __future__ import annotations

import sys
import functools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import recommendation_core as rc  # noqa: E402
import llm_local  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESOLVED_PATH = DATA_DIR / "Inc_resolved by MAIT.xlsx"
ESCALATED_PATH = DATA_DIR / "Inc_registred by MAIT.xlsx"
TICKET_EMBEDDINGS_PATH = DATA_DIR / "ticket_embeddings.npy"

TEST_QUERIES = [
    "GCA printer is not working and delivery sheets cannot be printed.",
    "MA - EXP - RESET PASSWORD LDAP : zelhadda. New account / modification / removal: Modification of password",
]


def main():
    print("Chargement de la base de connaissance...")
    kb = rc.load_knowledge_base(str(RESOLVED_PATH), str(ESCALATED_PATH))
    kb = rc.add_solution_types(kb)

    print("Chargement du modèle d'embeddings...")
    embed_model = rc.get_model()

    import numpy as np
    if TICKET_EMBEDDINGS_PATH.exists():
        ticket_embeddings = np.load(TICKET_EMBEDDINGS_PATH)
        if ticket_embeddings.shape[0] != len(kb):
            raise RuntimeError(
                "ticket_embeddings.npy désaligné avec la base actuelle -- "
                "relancez scripts/03_backtest.py une fois pour le régénérer."
            )
    else:
        raise FileNotFoundError(
            f"{TICKET_EMBEDDINGS_PATH} introuvable -- lancez d'abord "
            "scripts/03_backtest.py une fois (il génère les embeddings)."
        )

    print("Chargement du LLM local (premier appel : téléchargement, patience)...")
    llm_pipe = llm_local.get_llm_pipeline()
    llm_generator = functools.partial(llm_local.generate_recommendation_llm, pipe=llm_pipe)

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print("TICKET :", query.strip())
        print("=" * 80)

        result_heuristic = rc.recommend_incident(
            query=query,
            knowledge_base=kb,
            ticket_embeddings=ticket_embeddings,
            model=embed_model,
            # recommendation_generator omis -> heuristique par défaut
        )

        result_llm = rc.recommend_incident(
            query=query,
            knowledge_base=kb,
            ticket_embeddings=ticket_embeddings,
            model=embed_model,
            recommendation_generator=llm_generator,
        )

        print(f"\nStatus : {result_heuristic['status']}")

        print("\n--- RECOMMANDATION HEURISTIQUE (mots-clés) ---")
        print(result_heuristic.get("recommendation"))

        print("\n--- RECOMMANDATION LLM LOCAL ---")
        print(result_llm.get("recommendation"))


if __name__ == "__main__":
    main()