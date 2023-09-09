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

# TODO: Add Q-Learning (Reinforcement Learning model here)

def main():
    ticker = 'AAPL'
    look_back = 1
    data = fetch_data(ticker, '2020-01-01', '2021-01-01')
    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], look_back)
    X_test_flat = X_test.reshape(X_test.shape[0], look_back)

    model_type = input("Enter the model type (lr, logr, nb, rf, lstm, rnn): ")

    if model_type == 'lr':
        linear_regression_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'logr':
        logistic_regression_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'nb':
        naive_bayes_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'rf':
        random_forest_model(X_train_flat, Y_train, X_test_flat, Y_test)
    elif model_type == 'lstm':
        lstm_model(X_train, Y_train, X_test, Y_test, look_back)
    elif model_type == 'rnn':
        rnn_model(X_train, Y_train, X_test, Y_test, look_back)

if __name__ == "__main__":
    main()
