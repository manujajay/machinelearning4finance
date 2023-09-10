# Machine Learning Models in Finance

This repository contains various machine learning and deep learning models applicable to the financial domain.

## Table of Contents

- [Models Included](#models-included)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Data Fetching](#data-fetching)
- [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
- [Models Explained](#models-explained)

## Models Included

The repository consists of the following categories:

1. **Supervised Learning Models**
    - Linear Regression
    - Logistic Regression
    - Naive Bayes
    - Random Forest
2. **Unsupervised Learning Models**
    - Clustering (K-means)
    - Dimensionality Reduction (PCA)
3. **Deep Learning Models**
    - Recurrent Neural Networks (LSTM)
    - Convolutional Neural Networks (CNN)
    - Autoencoders
    - Generative Adversarial Networks (GANs)
4. **Reinforcement Learning Models**
    - Q-Learning

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
Data is fetched using the yfinance library for real-world financial data.

```python
import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values
```

## Data Preprocessing

Data is preprocessed to create training and testing datasets, which are then fed into machine learning models.

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

Navigate to the respective folder and run the Python script for the model you're interested in.

```bash
python script_name.py
```

## Models Explained

### Supervised Learning Models

![Linear Regression](./1.supervised_learning_models/linear_regression_summary_with_explanation.png)
#### Linear Regression
Linear Regression tries to fit a linear equation to the data, providing a straightforward and effective method for simple predictive tasks.

![Logistic Regression](./1.supervised_learning_models/logistic_regression_summary_with_explanation.png)
#### Logistic Regression
Logistic Regression is traditionally used for classification problems but has been adapted here for regression tasks.

![Naive Bayes](./1.supervised_learning_models/naive_bayes_summary_with_explanation.png)
#### Naive Bayes
Naive Bayes is particularly useful when you have a small dataset and is based on Bayes' theorem.

![Random Forest](./1.supervised_learning_models/random_forest_summary_with_explanation.png)
#### Random Forest
Random Forest combines multiple decision trees to make a more robust and accurate prediction model.

### Unsupervised Learning Models

![K-means](./2.unsupervised_learning_models/kmeans_financial_data_with_explanation.png)
#### Clustering (K-means)
K-means clustering is used to partition data into groups based on feature similarity.

![PCA](./2.unsupervised_learning_models/PCA_financial_data_with_full_explanation.png)
#### Dimensionality Reduction (PCA)
PCA is used to reduce the number of features in a dataset while retaining the most relevant information.

### Deep Learning Models

![Autoencoders](./3.deep_learning_models/Anomaly_Detection_Using_Autoencoder.png)
#### Autoencoders
Autoencoders are used for anomaly detection in financial data, identifying unusual patterns that do not conform to expected behavior.

![GANs](./3.deep_learning_models/GAN_Financial_Simulation.png)
#### Generative Adversarial Networks (GANs)
GANs are used for simulating different market conditions, helping in risk assessment for various investment strategies.

### Reinforcement Learning Models

![Q-Learning](./4.reinforcement_learning_models/Q_Learning_Stock_Trading_YFinance.png)
#### Q-Learning
Q-Learning is a type of model-free reinforcement learning algorithm used here for stock trading.

## Disclaimer

The code provided in this repository is for educational and informational purposes only. It is not intended for live trading or as financial advice. Please exercise caution and conduct your own research before making any investment decisions.
