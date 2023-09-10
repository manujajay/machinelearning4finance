# Import necessary libraries
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
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

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

# Analyze the components
components_df = pd.DataFrame(pca.components_, columns=data.columns, index=[f'PC{i+1}' for i in range(2)])

# Create plot with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Main plot on ax1
ax1.scatter(principal_components[:, 0], principal_components[:, 1], c='blue')
ax1.set_title('PCA of Stock Data and S&P 500')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')

# Turn off axis for ax2
ax2.axis('off')

# Initial explanation
initial_explanation = (
    "Algorithm: Principal Component Analysis (PCA)\n"
    "Components: 2\n"
    "Data: Stock prices and S&P 500 index\n\n"
    "PCA reduces the dimensionality of the data by finding new variables (Principal Components) that maximize variance."
    "This is useful for:\n"
    "- Data Visualization: Reducing dimensions aids in visualizing complex data.\n"
    "- Risk Modeling: Identifying primary risk factors in a portfolio.\n"
    "- Factor Analysis: Understanding the underlying factors affecting asset prices.\n"
)

# New section explaining the drivers of the components
new_section = (
    f"\nPrincipal Component 1 is most influenced by {components_df.loc['PC1'].idxmax()}.\n"
    f"Principal Component 2 is most influenced by {components_df.loc['PC2'].idxmax()}.\n"
)

# Combine initial explanation and new section
full_explanation = initial_explanation + new_section

# Additional useful explanation
why_useful = (
    "\nUsefulness:\n"
    "1. Portfolio Optimization: Identify key drivers of asset returns.\n"
    "2. Risk Management: Uncover main risk factors.\n"
    "3. Trading Strategies: Develop strategies based on hidden factors.\n"
    "4. Data Visualization: Easier interpretation of high-dimensional data.\n"
    "5. Correlation Structure: Simplify data complexity.\n"
    "6. Market Regime Identification: Adapt trading strategies dynamically."
)

# Combine the original explanation, the 'why useful' section, and the new section
full_explanation += why_useful

wrapped_full_explanation = "\n".join(wrap(full_explanation, 50))
ax2.text(0.01, 0.99, wrapped_full_explanation, fontsize=10, va='top', wrap=True)

# Save and show plot
plt.tight_layout()
plt.savefig('PCA_financial_data_with_full_explanation.png')
plt.show()

# Output the components for further analysis
print("Principal Component Analysis")
print(components_df)
