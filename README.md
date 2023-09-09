# Machine Learning Models in Finance

This repository contains Python scripts that demonstrates various machine learning and deep learning models for stock price prediction.

## Table of Contents

- [Models Included](#models-included)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Data Fetching](#data-fetching)
- [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
- [Models Explained](#models-explained)

## Models Included

The script includes the following models:

- 1. Logistic Regression
- 2. Linear Regression
- 3. Naive Bayes
- 4. Random Forest
- 5. Simple LSTM (Long Short-Term Memory)
- 6. Simple RNN (Recurrent Neural Network)
- 7. Q-Learning (Basic form of reinforcement learning)

## Dependencies

- Python 3.x
- yfinance
- NumPy
- TensorFlow
- Scikit-learn

## Installation

To install all dependencies, run:

```bash
pip install yfinance numpy tensorflow scikit-learn
```

## Data Fetching
The script fetches data using the yfinance library. It downloads the closing stock prices for a given ticker between specified dates.

```python
import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values
```

## Data Preprocessing
Data is preprocessed to create training and testing datasets. A function called create_dataset converts time-series stock data into a format suitable for machine learning.

```python
import numpy as np

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        X.append(a)
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)
```

## Usage
Run the script and input the model type you wish to use for stock prediction when prompted.

```bash
python app.py
```

## Models Explained

### Linear Regression
Linear Regression tries to fit a linear equation to the data. It's straightforward and good for simple tasks.

```python
from sklearn.linear_model import LinearRegression

def linear_regression_model(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    ...
```

### Logistic Regression
Logistic Regression is generally used for classification problems, but in this case, it has been adapted for regression.

```python
from sklearn.linear_model import LogisticRegression

def logistic_regression_model(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train.astype('int'))
    ...
```

### Naive Bayes
Naive Bayes is based on Bayes' theorem and is particularly good when you have a small dataset.

```python
from sklearn.naive_bayes import GaussianNB

def naive_bayes_model(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train.astype('int'))
    ...
```

### Random Forest
Random Forest combines multiple trees to predict the class (or in our case, the stock price).

```python
from sklearn.ensemble import RandomForestRegressor

def random_forest_model(X_train, Y_train, X_test, Y_test):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    ...
```

### Simple LSTM
LSTM (Long Short-Term Memory) models are a type of RNN that is good for sequence prediction problems.

```python
import tensorflow as tf

def lstm_model(X_train, Y_train, X_test, Y_test, look_back):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(look_back, 1)),
        tf.keras.layers.Dense(1)
    ])
    ...
```

### Simple RNN
RNN (Recurrent Neural Networks) are also good for sequence prediction problems but are generally less powerful than LSTMs.

```python
def rnn_model(X_train, Y_train, X_test, Y_test, look_back):
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(50, input_shape=(look_back, 1)),
        tf.keras.layers.Dense(1)
    ])
    ...
```

## Note
This is a working project and doesn't cover advanced features like hyperparameter tuning, feature selection, or advanced training regimes etc yet (I'm working on it).
Reinforcement learning models are also not fully covered in the first iteration - I will add that as I go.

## Disclaimer

The code provided in this repository is for educational and informational purposes only. It is not intended for live trading or investment advice. While every effort has been made to present accurate information, the code and its author make no representations or warranties as to the accuracy, reliability, or completeness of the information provided. Users of this code assume all risks associated with its usage, and the author accepts no responsibility or liability for any losses incurred. Please exercise caution and conduct your own research before making any investment decisions.
