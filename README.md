# Machine Learning Models in Finance

This repository contains various machine learning and deep learning models applicable to the financial domain.

## Table of Contents

- [1. Models Included](#models-included)
- [2. Dependencies](#dependencies)
- [3. Installation](#installation)
- [4. Data Fetching](#data-fetching)
- [5. Data Preprocessing](#data-preprocessing)
- [6. Usage](#usage)
- [7. Models Explained](#models-explained)
- [8. Beyond Core Machine Learning Models](#beyond-core-machine-learning-models-real-world-applications-in-finance)
- [8. Disclaimer](#disclaimer)

## 1. Models Included

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

## 2. Dependencies

- Python 3.x
- yfinance
- NumPy
- TensorFlow
- Scikit-learn

## 3. Installation

To install all dependencies, run:

```bash
pip install yfinance numpy tensorflow scikit-learn
```

## 4. Data Fetching
Data is fetched using the yfinance library for real-world financial data.

```python
import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values
```

## 5. Data Preprocessing

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

## 6. Usage

Navigate to the respective folder and run the Python script for the model you're interested in.

```bash
python script_name.py
```

## 7. Models Explained

### 1. Supervised Learning Models

#### 1.1 Linear Regression
Linear Regression tries to fit a linear equation to the data, providing a straightforward and effective method for simple predictive tasks.
![Linear Regression](./1.%20supervised_learning_models/linear_regression_summary_with_explanation.png)

#### 1.2 Logistic Regression
Logistic Regression is traditionally used for classification problems but has been adapted here for regression tasks.
![Logistic Regression](./1.%20supervised_learning_models/logistic_regression_summary_with_explanation.png)

#### 1.3 Naive Bayes
Naive Bayes is particularly useful when you have a small dataset and is based on Bayes' theorem.
![Naive Bayes](./1.%20supervised_learning_models/naive_bayes_summary_with_explanation.png)

#### 1.4 Random Forest
Random Forest combines multiple decision trees to make a more robust and accurate prediction model.
![Random Forest](./1.%20supervised_learning_models/random_forest_summary_with_explanation.png)

### 2. Unsupervised Learning Models

#### 2.1 Clustering (K-means)
K-means clustering is used to partition data into groups based on feature similarity.
![K-means](./2.%20unsupervised_learning_models/kmeans_financial_data_with_explanation.png)

#### 2.2 Dimensionality Reduction (PCA)
PCA is used to reduce the number of features in a dataset while retaining the most relevant information.
![PCA](./2.%20unsupervised_learning_models/PCA_financial_data_with_full_explanation.png)

### 3. Deep Learning Models

#### 3.1 Autoencoders
Autoencoders are used for anomaly detection in financial data, identifying unusual patterns that do not conform to expected behavior.
![Autoencoders](./3.%20deep_learning_models/Anomaly_Detection_Using_Autoencoder.png)

#### 3.2 Generative Adversarial Networks (GANs)
GANs are used for simulating different market conditions, helping in risk assessment for various investment strategies.
![GANs](./3.%20deep_learning_models/GAN_Financial_Simulation.png)

### 4. Reinforcement Learning Models

#### 4.1 Q-Learning
Q-Learning is a type of model-free reinforcement learning algorithm used here for stock trading.
![Q-Learning](./4.%20reinforcement_learning_models/Q_Learning_Stock_Trading_YFinance.png)

## 8. Beyond Core Machine Learning Models: Real-World Applications in Finance

In addition to the core machine learning models that form the backbone of this repository, we'll explore practical applications that span various dimensions of the financial sector. Below is a snapshot of the project's tree structure that gives you an idea of what these applications are:

```
5. ml_applications_in_finance
│   ├── risk_management
│   ├── decentralized_finance_(DEFI)
│   ├── environmental_social_and_governance_investing_(ESG)
│   ├── behavioural_economics
│   ├── blockchain_and_cryptocurrency
│   ├── explainable_AI_for_finance
│   ├── robotic_process_automation_(RPA)
│   ├── textual_and_alternative_data_for_finance
│   ├── fundamental_analysis
│   ├── satellite_image_analysis_for_finance
│   ├── venture_capital
│   ├── asset_management
│   ├── private_equity
│   ├── investment_banking
│   ├── trading
│   ├── portfolio_management
│   ├── wealth_management
│   ├── multi_asset_risk_model
│   ├── personal_financial_management_app
│   ├── market_analysis_and_prediction
│   ├── customer_service
│   ├── compliance_and_regulatory
│   ├── real_estate
│   ├── supply_chain_finance
│   ├── invoice_management
│   └── cash_management
```

From risk management to blockchain and cryptocurrency, from venture capital to investment banking, and from asset management to personal financial management, we aim to cover a wide array of use-cases. Each of these applications is backed by one or more of the machine learning models described earlier in the repository.

**Note**: The list of applications is not exhaustive, and the project is a work in progress. While I aim to continually update it with new techniques and applications, there might be instances where certain modules may be added or removed based on their relevance and effectiveness.


## Disclaimer

The code provided in this repository is for educational and informational purposes only. It is not intended for live trading or as financial advice. Please exercise caution and conduct your own research before making any investment decisions.
