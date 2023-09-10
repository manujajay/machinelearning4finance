import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Download stock data from Yahoo Finance
data = yf.download("AAPL", start="2021-01-01", end="2021-02-01")["Close"]
stock_prices = data.values
dates = data.index

# Initialize Q-Learning parameters
INITIAL_BALANCE = 1000.0
N_TRADING_DAYS = len(stock_prices)

# Initialize Q-Table
q_table = np.zeros(3)  # Buy, Sell, Hold

# Initialize records for rewards and balances
balances = [INITIAL_BALANCE]

# Initialize records for actions
actions = []

# Hyperparameters
epsilon = 0.2  # Exploration vs Exploitation
lr = 0.1  # Learning rate
gamma = 0.99  # Discount factor

# Simulation
for day in range(N_TRADING_DAYS - 1):
    state = balances[-1]
    stock_price = stock_prices[day]
    next_stock_price = stock_prices[day + 1]

    # Epsilon-greedy action selection
    action = np.random.randint(3) if np.random.rand() < epsilon else np.argmax(q_table)

    # Reward function
    reward = 0
    if action == 0:  # Buy
        reward = next_stock_price - stock_price
    elif action == 1:  # Sell
        reward = stock_price - next_stock_price

    # Q-Learning Update
    next_state = state + reward
    q_table[action] = q_table[action] + lr * (reward + gamma * np.max(q_table) - q_table[action])

    # Record action and balance
    actions.append(action)
    balances.append(next_state)

# Generate Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Stock prices with Buy/Sell markers
buy_dates = [dates[i] for i in range(len(actions)) if actions[i] == 0]
sell_dates = [dates[i] for i in range(len(actions)) if actions[i] == 1]

ax1.plot(dates, stock_prices, label='Stock Price')
ax1.scatter(buy_dates, [stock_prices[i] for i in range(len(actions)) if actions[i] == 0], marker='^', color='g', label='Buy', zorder=5)
ax1.scatter(sell_dates, [stock_prices[i] for i in range(len(actions)) if actions[i] == 1], marker='v', color='r', label='Sell', zorder=5)

ax1.set_title("Backtest with Buy/Sell Indicators")
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price")
ax1.legend()

# Balances
ax2.plot(dates, [INITIAL_BALANCE] + balances[:-1])
ax2.set_title("Balance Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Balance")

# Description and Statistics
statistics = f"""Parameters:
- Number of trading days: {N_TRADING_DAYS}
- Learning rate: {lr}
- Discount factor: {gamma}

Statistics:
- Final Balance: {balances[-1]:.2f}
"""

model_description = """This Q-learning model simulates stock trading decisions. 
It decides whether to buy, sell, or hold based on the history of stock prices. 
Green markers (^) indicate buying points, and red markers (v) indicate selling points."""

fig.text(0.2, 0.70, statistics, fontsize=7)
fig.text(0.65, 0.70, model_description, fontsize=6)

plt.suptitle("Stock Trading Simulation using Q-Learning with Yahoo Finance Data")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save plot
plt.savefig("Q_Learning_Stock_Trading_YFinance.png")

plt.show()
