"""
00_build_embeddings.py
======================

Reconstruit les embeddings alignés avec la base de connaissance
(tickets >= janvier 2025, solutions nettoyées).

À relancer chaque fois que :
  - les fichiers Excel changent,
  - ou que la logique de nettoyage / de filtre date change.

Usage (depuis la racine du projet) :
    python scripts/00_build_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import recommendation_core as rc  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESOLVED_PATH = DATA_DIR / "Inc_resolved by MAIT.xlsx"
ESCALATED_PATH = DATA_DIR / "Inc_registred by MAIT.xlsx"


def main() -> None:
    kb = rc.load_knowledge_base(str(RESOLVED_PATH), str(ESCALATED_PATH))
    kb = rc.add_solution_types(kb)

    print(f"Tickets dans la base : {len(kb)}")
    print(kb["solution_type"].value_counts().to_string())

    model = rc.get_model()

    ticket_texts = kb["text_for_embedding"].fillna("").astype(str).tolist()
    solution_texts = kb["Solution"].fillna("").astype(str).tolist()

    print("\nCalcul des embeddings des tickets...")
    ticket_embeddings = rc.compute_embeddings(ticket_texts, model=model)

    print("Calcul des embeddings des solutions...")
    solution_embeddings = rc.compute_embeddings(solution_texts, model=model)

    assert ticket_embeddings.shape[0] == len(kb)
    assert solution_embeddings.shape[0] == len(kb)

    np.save(DATA_DIR / "ticket_embeddings.npy", ticket_embeddings)
    np.save(DATA_DIR / "solution_embeddings.npy", solution_embeddings)

    print(f"\nOK -> ticket_embeddings    {ticket_embeddings.shape}")
    print(f"OK -> solution_embeddings  {solution_embeddings.shape}")


if __name__ == "__main__":
    main()