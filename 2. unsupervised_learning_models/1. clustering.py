# k-means clustering
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from textwrap import wrap

# Download stock data from Yahoo Finance
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
stock_data = yf.download(tickers, start='2020-01-01', end='2021-01-01')['Adj Close']

# Download S&P 500 data from FRED
sp500 = web.DataReader('SP500', 'fred', '2020-01-01', '2021-01-01')

# Combine stock and S&P 500 data
data = pd.concat([stock_data, sp500], axis=1).dropna()
data = data.pct_change().dropna()  # Calculate daily returns

# K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels = kmeans.labels_

# Create plot with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Main plot on ax1
ax1.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
ax1.set_title('K-Means Clustering of Stock Data and S&P 500')
ax1.set_xlabel('AAPL Daily Returns')
ax1.set_ylabel('GOOGL Daily Returns')

# Explanation on ax2
explanation = (
    "Algorithm: K-Means Clustering\n"
    "Number of Clusters: 3\n"
    "Data: Stock prices and S&P 500 index\n\n"
    "Explanation:\n"
    "K-means partitions the financial data into 'K' clusters based on daily returns. "
    "While it's not generally used for prediction, it provides valuable insights into data structure. "
    "These insights can be instrumental for:\n\n"
    "- Portfolio Diversification: Identifying statistically similar assets for diversification.\n"
    "- Risk Management: Recognizing asset groups for better hedging strategies.\n"
    "- Market Regime Identification: Understanding different market states for dynamic trading."
)
wrapped_explanation = "\n".join(wrap(explanation, 50))  # Wraps the text at 50 characters

ax2.axis('off')
ax2.text(0.01, 0.99, wrapped_explanation, fontsize=10, va='top', wrap=True)  # Aligns text at top left corner with a fontsize of 10

# Save and show plot
plt.tight_layout()
plt.savefig('kmeans_financial_data_with_explanation.png')
plt.show()
