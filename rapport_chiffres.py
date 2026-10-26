import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

d = pd.read_csv("data/backtest_results.csv")
d = d[d.Status == "SUCCESS"].copy()

d["Confiance"] = pd.cut(
    d.Ticket_Similarity, [0, 0.70, 0.80, 1.5],
    labels=["LOW (<0.70)", "MEDIUM (0.70-0.80)", "HIGH (>=0.80)"],
    right=False,
)

print("\nQualité par niveau de confiance :")
print(d.groupby("Confiance", observed=True).agg(
    tickets=("Test_Ticket", "count"),
    sol_sim_pipeline=("Solution_Similarity", "mean"),
    sol_sim_baseline=("Baseline_Top1_Solution_Similarity", "mean"),
    part_bonne=("Solution_Similarity", lambda s: (s >= 0.5).mean()),
).round(3))

# 30 cas à noter à la main : 0 = inutile, 1 = partiellement utile, 2 = bonne piste
echant = d.sample(30, random_state=7)[[
    "Test_Ticket", "Confiance", "Ticket_Similarity",
    "True_Solution", "Representative_Solution",
]]
echant["Note_manuelle_0_1_2"] = ""
echant.to_csv("data/evaluation_manuelle.csv", index=False, encoding="utf-8-sig")
print("\nOK -> data/evaluation_manuelle.csv (à noter dans Excel)")