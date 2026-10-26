import sys
sys.path.insert(0, "src")
import pandas as pd
import recommendation_core as rc

kb = rc.add_solution_types(
    rc.load_knowledge_base(
        "data/Inc_resolved by MAIT.xlsx",
        "data/Inc_registred by MAIT.xlsx",
    )
)
bt = pd.read_csv("data/backtest_results.csv")

m = bt.merge(
    kb[["Number", "solution_type"]].drop_duplicates("Number"),
    left_on="Test_Ticket", right_on="Number", how="left",
)

print("\nPipeline par type de VRAIE solution :")
print(m.groupby("solution_type").Solution_Similarity
        .agg(["count", "mean", "median"]).round(3))

print("\nBaseline par type de vraie solution :")
print(m.groupby("solution_type").Baseline_Top1_Solution_Similarity
        .mean().round(3))