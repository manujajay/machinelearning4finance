'''
Long Short-Term Memory networks (LSTMs) are a type of Recurrent Neural Network (RNN) and are typically used in the context
of supervised learning, particularly for sequence prediction problems like time series forecasting, natural language processing,
and more. In these applications, you usually have labeled data where the sequence input is associated with a corresponding output.

That being said, LSTMs can also be used in unsupervised learning scenarios. For example, you can use LSTMs in autoencoders for
sequence-to-sequence reconstruction, anomaly detection in time series data, or learning embeddings for sequences without explicit labels.
'''

'''
What is an LSTM?
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture. An LSTM is designed to remember past information in sequence data and is widely used in time series analysis, natural language processing, and many other sequence-related tasks. Unlike standard feedforward neural networks, LSTMs have "memory" in the form of a cell state and hidden state, which helps them learn from the "context" or "sequence" of the inputs.

How it works?
Input Sequence: At each time step, the LSTM takes in an input and the previous cell state and hidden state 

Forget Gate: Decide what information from the cell state should be thrown away.

Input Gate: Update the cell state with new information.

Output Gate: Based on the cell state and the input, decide what should be the new hidden state 

New Cell State: Finally, calculate the new cell state 

Predictive Power
LSTMs are particularly useful for solving problems that require learning long-term dependencies. They are less susceptible to the vanishing gradient problem, which allows them to learn from data where the important features are separated by many time steps. This makes them highly efficient for various sequence-based tasks such as time-series prediction, sequence-to-sequence mapping, and so on.

In finance, LSTMs can be used for predicting stock prices, forex trading, and even for algorithmic trading strategies. However, it's crucial to note that the financial markets are influenced by a multitude of factors, many of which can be non-sequential or not included in the model. So while LSTMs can capture patterns in past data efficiently, they are by no means a guarantee for high accuracy in financial predictions.

By setting up a proper evaluation metric (like RMSE for regression tasks, or F1-score for classification tasks), you can get a quantitative measure of how well your LSTM model is likely to perform on unseen data.
'''
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

# Download the Apple stock price data
data = yf.download('AAPL', start='2019-01-01', end='2021-01-01')
data = data[['Close']]

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Create a dataset for training the LSTM model
train_data = scaled_data[:int(0.8 * len(scaled_data))]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Debugging
print(f"Total data length: {len(data)}")
print(f"Training data length: {len(train_data)}")

# Building and Training the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, batch_size=1, epochs=1)

# Output the training loss
print(f"Training loss: {history.history['loss'][0]}")

# Testing the Model
test_data = scaled_data[int(0.8 * len(scaled_data)) - 60:]
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the test set
predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(np.reshape(predicted_price, (-1, 1)))

# Calculate Test Loss
test_loss = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")

# Calculate Root Mean Square Error (RMSE)
rmse = sqrt(mean_squared_error(y_test, predicted_price))
print(f'Root Mean Square Error (RMSE): {rmse}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predicted_price)
print(f'Mean Absolute Error (MAE): {mae}')

# Visualizing the Results
plt.figure(figsize=(16, 8))

# Plot the real stock price
plt.plot(data.index, data['Close'], label='True Price')

# Generate the index for the predicted price
predicted_index = data.index[-100:]  # Adjust the number to match the new shape

# Debugging: Verifying dimensions before plotting
print(f"Shape of predicted_price: {predicted_price.shape}")
print(f"Shape of predicted_index: {len(predicted_index)}")
print(f"First few elements of predicted_index: {predicted_index[:5]}")
print(f"Last few elements of predicted_index: {predicted_index[-5:]}")
print(f"Length of data.index: {len(data.index)}")
print(f"Length of train_data: {len(train_data)}")
print(f"Length of train_data + 60: {len(train_data) + 60}")


# Adjusting predicted_price to match the length of predicted_index
predicted_price = predicted_price[:len(predicted_index)]

# Plot the predicted stock price
if len(predicted_index) == predicted_price.shape[0]:
    plt.plot(predicted_index, predicted_price.flatten(), label='Predicted Price')
else:
    print("Shape mismatch: Skipping plotting of predicted prices")

# Add performance metrics and explanations to the plot
metrics_text = f'''Test Loss: {0.0194} (Lower is better)
RMSE: {109.8886} (Lower is better, dependent on scale of target variable)
MAE: {109.8511} (Lower is better, dependent on scale of target variable)'''

plt.text(0.02, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='center', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

plt.legend()
plt.title("Apple Stock Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("Stock Price")

# Save the plot as a .png file
plt.savefig('Apple_Stock_Price_Prediction.png')

# Show the plot
plt.show()
