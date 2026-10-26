import pandas as pd

d = pd.read_csv("data/backtest_results.csv")
d = d[d.Status == "SUCCESS"]
faibles = d[d.Solution_Similarity < 0.30].sample(20, random_state=1)

cols = ["Test_Ticket", "Ticket_Similarity", "Solution_Similarity",
        "True_Solution", "Representative_Solution"]
faibles[cols].to_csv("data/cas_faibles.csv", index=False, encoding="utf-8-sig")
print("OK -> data/cas_faibles.csv")