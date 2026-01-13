import pandas as pd
import matplotlib.pyplot as plt

# Load your backtest results
df = pd.read_csv("database\csv\equity_curve_from_candles.csv", parse_dates=["ts"])

# Sort by timestamp just in case
df = df.sort_values("ts")

# Plot balance over time
plt.figure(figsize=(12,6))
plt.plot(df["ts"], df["balance"], label="Equity Curve", linewidth=1.5)

plt.title("Backtest Equity Curve", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Balance", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
