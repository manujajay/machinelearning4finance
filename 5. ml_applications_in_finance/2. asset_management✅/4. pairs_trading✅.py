# Identifying pairs of assets whose prices have a statistical relationship, used for arbitrage

'''
Pairs trading is a strategy that identifies pairs of assets (typically stocks) that are historically price correlated.
When their prices deviate substantially, one stock is shorted while the other is bought, with the expectation that the two prices will converge again.

Here's a basic outline for a Pairs Trading strategy:

1. Data Collection: Fetch historical data for a set of potential pairs.
2. Pair Selection: Identify pairs with a strong statistical relationship.
3. Signal Generation: Determine entry (long/short) and exit points based on a Z-score of the spread.
4. Trade Execution & Management: Execute the trades and manage the positions.
'''

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# 1. Data Collection
tickers = ['DAL', 'AAL', 'UAL', 'LUV']  # airline stocks as an example
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
data = data.fillna(method='ffill')  # forward-fill missing values

# 2. Pair Selection
def find_cointegrated_pairs(data, pvalue_threshold=0.1):  # Adjusted threshold
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < pvalue_threshold:  # P-value threshold
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

_, _, pairs = find_cointegrated_pairs(data)
print("Cointegrated pairs:", pairs)

# Check if we have any cointegrated pairs before proceeding
if not pairs:
    print("No cointegrated pairs found!")
    exit()

# For demonstration, let's use the first cointegrated pair.
S1 = data[pairs[0][0]]
S2 = data[pairs[0][1]]

# Calculate the spread
spread = S1 - S2
spread_mean = spread.mean()
spread_std = spread.std()

# 3. Signal Generation
zscore = (spread - spread_mean) / spread_std
entry_threshold = 1.5
exit_threshold = 0.5

# Buy when zscore < -entry_threshold, sell when zscore > -exit_threshold
# Short when zscore > entry_threshold, cover when zscore < exit_threshold
longs = (zscore < -entry_threshold) & (zscore.shift(1) > -entry_threshold)
shorts = (zscore > entry_threshold) & (zscore.shift(1) < entry_threshold)
exits = (np.abs(zscore) < exit_threshold)

# 4. Plotting
plt.figure(figsize=(15,7))

S1[longs].plot(marker='^', markersize=10, color='g', linestyle='None', alpha=0.7)
S1[shorts].plot(marker='v', markersize=10, color='r', linestyle='None', alpha=0.7)
S1[exits].plot(marker='o', markersize=6, color='b', linestyle='None', alpha=0.7)
S1.plot(color='b')
S2.plot(color='c')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Pairs Trading')
plt.legend([pairs[0][0], pairs[0][1], 'Buy Signal', 'Sell Signal', 'Exit Signal'])
plt.show()

