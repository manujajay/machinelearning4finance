import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from fredapi import Fred

# Fetch stock data
def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values

# Fetch economic indicators using the FRED API
def fetch_fred_data(api_key, series_id, start_date, end_date):
    fred = Fred(api_key=api_key)
    return fred.get_series(series_id, start_date, end_date).values

# Create dataset with predictors and target
def create_dataset(stock_data, sp500_data, interest_rates, gdp_growth, unemployment, look_back=1):
    X, Y = [], []
    for i in range(len(stock_data) - look_back - 1):
        features = list(stock_data[i:(i + look_back)]) + [sp500_data[i], interest_rates[i], gdp_growth[i], unemployment[i]]
        X.append(features)
        Y.append(1 if stock_data[i + look_back] > stock_data[i + look_back - 1] else 0)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # Get FRED API key from environment variable
    api_key = os.getenv('FRED_API_KEY')
    if api_key is None:
        print("Please set your FRED_API_KEY as an environment variable.")
        exit()

    # Fetch data
    stock_data = fetch_data('AAPL', '2010-01-01', '2023-01-01')
    sp500_data = fetch_data('^GSPC', '2010-01-01', '2023-01-01')
    interest_rates = fetch_fred_data(api_key, 'TB3MS', '2010-01-01', '2023-01-01')
    gdp_growth = fetch_fred_data(api_key, 'A191RL1Q225SBEA', '2010-01-01', '2023-01-01')
    unemployment = fetch_fred_data(api_key, 'UNRATE', '2010-01-01', '2023-01-01')

    # After fetching the data, ensure they all have the same length
    min_len = min(len(stock_data), len(sp500_data), len(interest_rates), len(gdp_growth), len(unemployment))

    stock_data = stock_data[:min_len]
    sp500_data = sp500_data[:min_len]
    interest_rates = interest_rates[:min_len]
    gdp_growth = gdp_growth[:min_len]
    unemployment = unemployment[:min_len]


    # Create dataset
    X, Y = create_dataset(stock_data, sp500_data, interest_rates, gdp_growth, unemployment)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, pred)
    cm = confusion_matrix(Y_test, pred)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='g', ax=ax1, cmap='Blues')
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title('Confusion Matrix')

    # Explanation text
    explanation = (
        f"Model: Random Forest\n"
        f"Accuracy: {accuracy:.2f}\n\n"
        "Predictors:\n"
        "- Previous day's stock price\n"
        "- S&P 500 index\n"
        "- Interest rates\n"
        "- GDP growth rates\n"
        "- Unemployment rates\n\n"
        "Explanation:\n"
        "The Random Forest model is an ensemble learning method that combines multiple decision trees to improve model performance. "
        "Here's how it works:\n"
        "1. Bagging: Random subsets of the training data are chosen to train each tree.\n"
        "2. Decision Trees: Each subset constructs a decision tree. Each node in these trees considers a random subset of features when splitting.\n"
        "3. Prediction: In classification tasks like this one, each tree's prediction is considered as a 'vote'. The class with the most votes is the final prediction.\n"
        "4. For Classification: Majority Voting is used to aggregate the 'votes' from all trees.\n"
        "Equation for Classification: y = mode(y1, y2, ..., yN), where N is the number of trees.\n"
    )

    ax2.axis('off')
    ax2.text(0.01, 0.99, explanation, fontsize=9, va='top')  # Aligns text at top left corner with a fontsize of 9

    plt.tight_layout()
    plt.savefig('random_forest_summary_with_explanation.png')
    plt.show()
