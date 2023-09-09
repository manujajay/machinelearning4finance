import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        X.append(a)
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

def linear_regression_model(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(Y_test, pred))

def logistic_regression_model(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train.astype('int'))
    pred = model.predict(X_test)
    print("Logistic Regression MSE:", mean_squared_error(Y_test, pred))

def naive_bayes_model(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train.astype('int'))
    pred = model.predict(X_test)
    print("Naive Bayes MSE:", mean_squared_error(Y_test, pred))

def random_forest_model(X_train, Y_train, X_test, Y_test):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print("Random Forest MSE:", mean_squared_error(Y_test, pred))

def lstm_model(X_train, Y_train, X_test, Y_test, look_back):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(look_back, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=2, batch_size=1)
    pred = model.predict(X_test)
    print("LSTM MSE:", mean_squared_error(Y_test, pred))

def rnn_model(X_train, Y_train, X_test, Y_test, look_back):
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(50, input_shape=(look_back, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=2, batch_size=1)
    pred = model.predict(X_test)
    print("RNN MSE:", mean_squared_error(Y_test, pred))

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((self.states, len(self.actions)))
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (target - predict)

def reinforcement_learning_q_learning(data, look_back=1):
    n_actions = 3  # Buy, Sell, Hold
    agent = QLearningAgent(len(data) - look_back, range(n_actions))
    state = 0
    for i in range(0, len(data) - look_back - 1):
        state = i
        action = agent.choose_action(state)
        next_state = state + 1
        # Here you can define your own reward function based on the action and price change
        reward = data[next_state] - data[state] if action == 0 else 0  # Simplified reward function
        agent.learn(state, action, reward, next_state)

    # Predict the last action based on Q-values. 
    # You can extend this part to make multiple predictions.
    final_state = len(data) - look_back - 1
    final_action = agent.choose_action(final_state)
    return final_action

def main():
    ticker = 'AAPL'
    look_back = 1
    data = fetch_data(ticker, '2020-01-01', '2021-01-01')
    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], look_back)
    X_test_flat = X_test.reshape(X_test.shape[0], look_back)

    model_type = input("Enter the model type (linear_regression, logistic_regression, naive_bayes, random_forest, lstm, rnn, reinforcement_learning_q_learning): ")

    if model_type == 'linear_regression':
        linear_regression_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'logistic_regression':
        logistic_regression_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'naive_bayes':
        naive_bayes_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'random_forest':
        random_forest_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'lstm':
        lstm_model(X_train, Y_train, X_test, Y_test, look_back)
    elif model_type == 'rnn':
        rnn_model(X_train, Y_train, X_test, Y_test, look_back)
    elif model_type == 'reinforcement_learning_q_learning':
        final_action = reinforcement_learning_q_learning(data, look_back)
        print(f"Final action suggested by Q-Learning: {['Buy', 'Sell', 'Hold'][final_action]}")

if __name__ == "__main__":
    main()
