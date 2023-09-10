import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])
        Y.append(1 if data[i + look_back] > data[i + look_back - 1] else 0)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # Fetch and prepare data
    data = fetch_data('AAPL', '2010-01-01', '2023-01-01')
    X, Y = create_dataset(data)
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
        "Explanation:\n"
        "Logistic Regression is a classification algorithm.\n"
        "In this context, it predicts whether the stock price will go up (1) or down (0) the next day.\n"
        "Accuracy is the metric used to evaluate the model's performance.\n"
        "The confusion matrix provides a summary of the number of correct and incorrect predictions."
    )

    ax2.axis('off')
    ax2.text(0.1, 0.1, explanation, fontsize=12)

    plt.tight_layout()
    plt.savefig('logistic_regression_summary_with_explanation.png')
    plt.show()
