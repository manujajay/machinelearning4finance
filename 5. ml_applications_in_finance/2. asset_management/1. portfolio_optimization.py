# Models to maximize returns for a given level of risk.

'''
Portfolio optimization using machine learning involves selecting the mix of investment assets that is statistically likely
to achieve a desired return for a given level of risk. One of the popular methods for this is the use of the
Mean-Variance Optimization model. Here, I'll show you a simplified example using Random Forest Regression to predict future returns,
and then optimizing the portfolio based on those returns.
'''

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pandas_datareader import data as pdr
import datetime
import concurrent.futures

# Fetch stock names using multi-threading
def get_stock_name(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get('shortName') or info.get('longName') or ticker

def fetch_all_stock_names(tickers):
    ticker_to_name = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ticker = {executor.submit(get_stock_name, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                ticker_to_name[ticker] = future.result()
            except Exception as e:
                print(f"Could not fetch name for {ticker}: {e}")
                ticker_to_name[ticker] = ticker
    return ticker_to_name

# Streamlit title and setup
st.title("Portfolio Optimization App")

# Timeframe
timeframe = st.selectbox('Select Timeframe:', ['1Y', '2Y', '3Y'])
start_date = str((datetime.datetime.now() - pd.DateOffset(years=int(timeframe[0]))).date())
end_date = str(datetime.datetime.now().date())

# Categorized Tickers
all_tickers = {
    'Stocks': ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    'Stock ETFs': ["SPY", "QQQ", "EFA", "IWM", "EEM"],
    'Commodities': ["GLD", "SLV", "USO"],
    'Bond ETFs': ["TLT"],
    'Real Estate ETFs': ["VNQ"],
    'Highlighted ETFs': ["0P0000XMRD.L", "0P0000KSPA.L", "0P000023MW.L", "0P000185T1.L", "0P0000TKZG.L"],
    'Highlighted Stocks': ["NTDOY", "PLTK", "INSE", "SCPL", "EA"]
}

all_tickers_flat = [item for sublist in all_tickers.values() for item in sublist]
ticker_to_name = fetch_all_stock_names(all_tickers_flat)

# Selection
selected_tickers = []
for category, tickers in all_tickers.items():
    st.write(f"## {category}")
    selected = st.multiselect('', tickers, format_func=lambda x: f"{ticker_to_name[x]} ({x})")
    selected_tickers.extend(selected)

# Make sure SPY is in the selected_tickers for benchmarking
if 'SPY' not in selected_tickers:
    selected_tickers.append('SPY')

# Download stock data
def download_data(ticker_list, start_date, end_date):
    data = yf.download(ticker_list, start=start_date, end=end_date)['Adj Close']
    return data.pct_change().dropna()

# Portfolio optimization
def optimize_portfolio(returns):
    def objective(weights):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return -portfolio_return / portfolio_volatility

    initial_weights = [1. / len(returns.columns)] * len(returns.columns)
    bounds = tuple((0, 1) for asset in range(len(returns.columns)))
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    solution = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    return solution.x

# Download data
data = download_data(selected_tickers, start_date, end_date)

# Optimize portfolio
optimal_weights = optimize_portfolio(data)

# Display Results
st.write("## Final Portfolio Allocation:")
allocation_df = pd.DataFrame({
    'Asset': [ticker_to_name[ticker] for ticker in selected_tickers],
    'Ticker': selected_tickers,
    'Weights': optimal_weights
})
st.table(allocation_df)

# Pie Chart (Only include assets with at least 1% weight)
significant_weights = optimal_weights[optimal_weights >= 0.01]
significant_tickers = np.array(selected_tickers)[optimal_weights >= 0.01]

st.write("## Portfolio Allocation Chart:")
fig, ax = plt.subplots()
ax.pie(significant_weights, labels=[ticker_to_name[ticker] for ticker in significant_tickers], autopct='%1.1f%%')
ax.axis('equal')
st.pyplot(fig)

# Performance vs. SPY
cumulative_portfolio_return = (data * optimal_weights).sum(axis=1).add(1).cumprod().sub(1)
cumulative_spy_return = data['SPY'].add(1).cumprod().sub(1)

fig, ax = plt.subplots()
cumulative_portfolio_return.plot(ax=ax, label='Portfolio')
cumulative_spy_return.plot(ax=ax, label='SPY')
plt.legend()
plt.title("Portfolio Performance vs. SPY")
st.pyplot(fig)

# Calculate Sharpe Ratio
risk_free_data = pdr.get_data_fred('GS3M', start_date, end_date)
risk_free_data_monthly = risk_free_data.resample('M').mean()
risk_free_data_monthly.interpolate(method='linear', inplace=True)
risk_free_data_aligned = risk_free_data_monthly.reindex(data.index, method='pad') / 100 / 252

portfolio_return = (data * optimal_weights).sum(axis=1)
excess_portfolio_return = portfolio_return.sub(risk_free_data_aligned['GS3M'].squeeze(), axis=0)

sharpe_ratio = np.sqrt(252) * (excess_portfolio_return.mean() / excess_portfolio_return.std())
st.write(f"## Annualized Sharpe Ratio: {sharpe_ratio:.4f}")


# Note: This is a simplified example. Always consult with a financial advisor before making any investment decisions.

# To run, do:
# "streamlit run 1.\ portfolio_optimization.py"



'''
Explanation:

Synthetic Data: The example uses synthetic stock return data. In a real-world scenario, you'd use historical stock return data.
UPDATE: Replaced with yfinance data.

User Input: The desired annual return and risk tolerance can be input by the user.

Data Splitting: The data is split into training and test sets. We use the training set to train our machine learning model.

Random Forest Regressor: We use Random Forest to predict the future returns of each asset in the test dataset.

Predictions and Plot: We make predictions using the trained models and plot the real vs. predicted returns of the first asset.

Optimization: Using predicted returns, we optimize the portfolio weights to maximize the Sharpe Ratio subject to the user-defined constraints on return and risk. This is done using the minimize function from scipy's optimize module.

Optimal Weights: Finally, the script prints the optimal weights for asset allocation to achieve the user's financial goals.
'''