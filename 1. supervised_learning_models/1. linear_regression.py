import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)['Close'].values

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # Fetch and prepare data
    data = fetch_data('AAPL', '2010-01-01', '2023-01-01')
    X, Y = create_dataset(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(Y_test, pred)
    r2 = r2_score(Y_test, pred)
    
    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))
    
    # Plotting the actual vs predicted values on first subplot
    ax[0].scatter(range(len(Y_test)), Y_test, c='g', label='Actual')
    ax[0].scatter(range(len(pred)), pred, c='r', label='Predicted')
    ax[0].set_xlabel('Index in Test Set')
    ax[0].set_ylabel('Stock Price (USD)')
    ax[0].legend()
    ax[0].set_title('Linear Regression Model: Actual vs Predicted Stock Prices')
    
    # Annotations and equations on the second subplot
    ax[1].axis('off')
    ax[1].text(0.1, 0.8, f'Model: Linear Regression', fontsize=12)
    ax[1].text(0.1, 0.7, f'Equation: Y = {model.coef_[0]:.2f} * X + {model.intercept_:.2f}', fontsize=12)
    ax[1].text(0.1, 0.6, f'Mean Squared Error: {mse:.2f}', fontsize=12)
    ax[1].text(0.1, 0.5, f'R^2 Score: {r2:.2f}', fontsize=12)
    
    # Explanation
    explanation = (
        "Explanation:\n"
        "Linear Regression tries to fit a linear equation to the data points.\n"
        "In this case, we are trying to predict the future stock price of Apple Inc.\n"
        "The model takes the stock price of a previous day (X) and predicts the stock price\n"
        "for the next day (Y) using the equation Y = Coefficient * X + Intercept.\n"
        "MSE and R^2 Score are metrics to evaluate the model's performance."
    )
    ax[1].text(0.1, 0.1, explanation, fontsize=12)

    # Save plot as a PNG file
    plt.savefig('linear_regression_summary_with_explanation.png')

    # Show the plot
    plt.show()
