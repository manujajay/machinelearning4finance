import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from fredapi import Fred

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values

def fetch_fred_data(api_key, series_id, start_date, end_date):
    fred = Fred(api_key=api_key)
    return fred.get_series(series_id, start_date, end_date).values

def create_dataset(stock_data, sp500_data, interest_rates, gdp_growth, unemployment, look_back=1):
    X, Y = [], []
    for i in range(len(stock_data) - look_back - 1):
        features = list(stock_data[i:(i + look_back)]) + [sp500_data[i], interest_rates[i], gdp_growth[i], unemployment[i]]
        X.append(features)
        Y.append(1 if stock_data[i + look_back] > stock_data[i + look_back - 1] else 0)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    api_key = os.getenv('FRED_API_KEY')
    if api_key is None:
        print("Please set your FRED_API_KEY as an environment variable.")
        exit()

    # Fetch and prepare data
    stock_data = fetch_data('AAPL', '2010-01-01', '2023-01-01')
    sp500_data = fetch_data('^GSPC', '2010-01-01', '2023-01-01')
    interest_rates = fetch_fred_data(api_key, 'TB3MS', '2010-01-01', '2023-01-01')
    gdp_growth = fetch_fred_data(api_key, 'A191RL1Q225SBEA', '2010-01-01', '2023-01-01')
    unemployment = fetch_fred_data(api_key, 'UNRATE', '2010-01-01', '2023-01-01')

    # Truncate data to the smallest length among all series
    min_len = min(len(stock_data), len(sp500_data), len(interest_rates), len(gdp_growth), len(unemployment))
    stock_data, sp500_data, interest_rates, gdp_growth, unemployment = [arr[:min_len] for arr in [stock_data, sp500_data, interest_rates, gdp_growth, unemployment]]

    X, Y = create_dataset(stock_data, sp500_data, interest_rates, gdp_growth, unemployment)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(Y_test, pred)
    cm = confusion_matrix(Y_test, pred)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confusion matrix using Seaborn
    sns.heatmap(cm, annot=True, fmt='g', ax=ax1, cmap='Blues')
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title('Confusion Matrix')

    # Explanation
    explanation = (
        f"Model: Logistic Regression\n"
        f"Accuracy: {accuracy:.2f}\n\n"
        "Predictors:\n"
        "- Previous day's stock price\n"
        "- S&P 500 index\n"
        "- Interest rates\n"
        "- GDP growth rates\n"
        "- Unemployment rates\n\n"
        "Explanation:\n"
        "Logistic Regression is a classification algorithm.\n"
        "It uses multiple predictors to estimate the probability of the stock price going up (1) or down (0) the next day.\n"
        "Accuracy is the metric used to evaluate the model's performance.\n"
        "The confusion matrix provides a summary of the number of correct and incorrect predictions."
    )

    ax2.axis('off')
    ax2.text(0.1, 0.1, explanation, fontsize=12)

    plt.tight_layout()
    plt.savefig('logistic_regression_summary_with_explanation.png')
    plt.show()

