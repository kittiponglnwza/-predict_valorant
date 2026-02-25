import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =====================================
# 1) Download Data
# =====================================

tickers = ["NVDA", "AMD"]
data = yf.download(tickers, start="2015-01-01", end="2024-12-31")["Close"]
data.dropna(inplace=True)

# =====================================
# 2) Compute Returns
# =====================================

data["Ret_NVDA"] = data["NVDA"].pct_change()
data["Ret_AMD"] = data["AMD"].pct_change()
data.dropna(inplace=True)

lookback = 60
rebalance = 20
target_vol = 0.20  # 20% annual

betas = []
positions = []

for i in range(len(data)):
    
    if i < lookback:
        betas.append(np.nan)
        positions.append(0)
        continue

    window = data.iloc[i-lookback:i]

    # Regression for beta
    X = sm.add_constant(window["Ret_AMD"])
    y = window["Ret_NVDA"]
    model = sm.OLS(y, X).fit()
    beta = model.params["Ret_AMD"]

    betas.append(beta)

    # Relative strength
    rs_nvda = data["NVDA"].pct_change(lookback).iloc[i]
    rs_amd = data["AMD"].pct_change(lookback).iloc[i]

    if i % rebalance == 0:
        if rs_nvda > rs_amd:
            pos = 1
        else:
            pos = -1

    positions.append(pos)

data["Beta"] = betas
data["Signal"] = positions

data.dropna(inplace=True)

# =====================================
# 3) Compute Raw Strategy Return
# =====================================

data["Raw_Return"] = (
    data["Signal"].shift(1) * (
        data["Ret_NVDA"] -
        data["Beta"] * data["Ret_AMD"]
    )
)

# =====================================
# 4) Volatility Scaling
# =====================================

rolling_vol = data["Raw_Return"].rolling(60).std() * np.sqrt(252)
data["Scaled_Return"] = data["Raw_Return"] * (target_vol / rolling_vol)

data.dropna(inplace=True)

data["Cumulative"] = (1 + data["Scaled_Return"]).cumprod()

# =====================================
# 5) Metrics
# =====================================

sharpe = (
    data["Scaled_Return"].mean() /
    data["Scaled_Return"].std()
) * np.sqrt(252)

max_dd = (
    data["Cumulative"] /
    data["Cumulative"].cummax() - 1
).min()

print("\n===== QUANT-GRADE RELATIVE STRENGTH =====")
print("Total Return:", data["Cumulative"].iloc[-1])
print("Sharpe:", sharpe)
print("Max Drawdown:", max_dd)

# =====================================
# 6) Plot
# =====================================

plt.figure(figsize=(12,6))
plt.plot(data.index, data["Cumulative"])
plt.title("Quant-Grade NVDA vs AMD")
plt.show()