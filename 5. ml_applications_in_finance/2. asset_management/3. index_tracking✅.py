# Algorithms to create a portfolio that closely follows a particular index.

'''
Creating a portfolio that tracks an index is the basis for index funds and ETFs. One way to do this is by finding the optimal weights of the stocks in the index such that the tracking error is minimized.

In this script, I'll outline the following steps:

1. Fetch the historical data of stocks in the index using yfinance.
2. Calculate the returns of each stock.
3. Define an optimization problem to find the optimal weights that minimize the tracking error.
4. Plot the actual index performance vs. our portfolio's performance.
'''

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Fetch Data
# Let's say we want to track the S&P 500. We'll take a subset of companies for simplicity.
tickers = ['AAPL', 'MSFT', 'GOOGL', '^GSPC']  # ^GSPC is the ticker for S&P 500
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']

# 2. Calculate Returns
returns = data.pct_change().dropna()

# 3. Optimization
def tracking_error(weights: np.array) -> float:
    # Calculate the portfolio returns given the weights
    port_returns = returns.iloc[:, :-1].dot(weights)
    # Calculate the tracking error
    error = np.sum((port_returns - returns['^GSPC'])**2)
    return error

# Constraints and bounds
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1) for _ in range(returns.shape[1] - 1)]  # the number of stocks minus one for the S&P 500 index column

# Minimize the negative Sharpe Ratio to get maximum Sharpe ratio
initial_guess = [1./len(tickers) for _ in tickers[:-1]]  # equal weights as an initial guess
result = minimize(tracking_error, initial_guess, bounds=bounds, constraints=cons)

# Extract the optimal weights
optimal_weights = result.x

# 4. Plot
# Calculate portfolio with optimal weights
data['Portfolio'] = data.iloc[:, :-1].dot(optimal_weights)
normalized_data = data / data.iloc[0]  # Normalize data for better visualization

plt.figure(figsize=(14, 7))
normalized_data['^GSPC'].plot(label='S&P 500')
normalized_data['Portfolio'].plot(label='Tracked Portfolio')
plt.title('Index Tracking')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
plt.show()

# Display optimal weights
print("Optimal Weights:", optimal_weights)
