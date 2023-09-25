# Utilizing algorithms and quantitative models to execute trades at optimal prices.
# Algorithmic Trading Script Using Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fetching data using yfinance
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# Feature Engineering

data['Close_Lag1'] = data['Close'].shift(1)
data['Return'] = (data['Close'] - data['Close_Lag1']) / data['Close_Lag1']
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA_Diff'] = data['MA5'] - data['MA10']
data['Momentum'] = data['Close'] - data['Close'].shift(4)
data['Volatility'] = data['Return'].rolling(window=5).std()
data.dropna(inplace=True)
data['Target'] = (data['Return'] > 0).astype(int)

features = ['Close', 'Close_Lag1', 'MA5', 'MA10', 'MA_Diff', 'Momentum', 'Volatility']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy [Random Forest]: {accuracy:.2f}")

data['RF_Predicted_Signal'] = np.nan
data.iloc[(len(data) - len(y_pred)):, data.columns.get_loc('RF_Predicted_Signal')] = y_pred
data['RF_Strategy_Return'] = data['Return'] * (data['RF_Predicted_Signal'] * 2 - 1)
data['RF_Cumulative_Strategy_Returns'] = (1 + data['RF_Strategy_Return']).cumprod()

data['Cumulative_Market_Returns'] = (1 + data['Return']).cumprod()

# Plot
fig, ax = plt.subplots(figsize=(15, 10))

# Plot stock prices
ax.plot(data.index, data['Close'], color='g', label='Stock Price', alpha=0.5)

# Plot strategy and market returns
test_data_start = data.iloc[len(data) - len(y_pred):].index[0]
ax.plot(data.loc[test_data_start:]['RF_Cumulative_Strategy_Returns'], color='b', label='Random Forest Strategy Returns')
ax.plot(data.loc[test_data_start:]['Cumulative_Market_Returns'], color='r', label='Buy and Hold Returns')
ax.legend(loc="upper left")
ax.set_ylabel('Value')

plt.tight_layout()
plt.show()
