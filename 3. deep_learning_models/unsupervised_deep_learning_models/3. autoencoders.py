import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Generate synthetic financial data (replace with real financial data)
X, _ = make_classification(n_samples=1000, n_features=20)
X = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Create autoencoder model
input_layer = Input(shape=(20,))
encoded = Dense(14, activation='relu')(input_layer)
decoded = Dense(20, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Use the trained autoencoder to predict test data
X_test_predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_predictions, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.quantile(mse, 0.95)

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# Plot histogram
axes[0].hist(mse, bins=50, alpha=0.6, color='g', label='Normal')
axes[0].axvline(x=threshold, color='r', linestyle='dashed', linewidth=2, label=f'Anomaly threshold ({threshold:.4f})')
axes[0].set_title("Anomaly Detection using Autoencoder in Finance")
axes[0].set_xlabel("Mean Squared Error (MSE)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Plot MSE over samples
axes[1].plot(mse, label='MSE')
axes[1].axhline(y=threshold, color='r', linestyle='dashed', linewidth=2, label=f'Anomaly threshold ({threshold:.4f})')
axes[1].scatter(np.where(mse > threshold), mse[mse > threshold], color='r', zorder=5, label='Anomalies')
axes[1].set_title("MSE Values Over Test Samples")
axes[1].set_xlabel("Test Sample Index")
axes[1].set_ylabel("Mean Squared Error (MSE)")
axes[1].legend()

# Add key statistics
stats_text = f'''Key Statistics:
- Training Data Size: {X_train.shape[0]}
- Test Data Size: {X_test.shape[0]}
- Anomaly Threshold (95 percentile): {threshold:.4f}'''

fig.text(0.15, 0.1, stats_text, fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

# Add model explanation
model_explanation = '''Model Explanation:
Autoencoders can be valuable in detecting anomalies in trading and identifying fraudulent transactions.
By training the autoencoder on 'normal' financial data, it learns to reconstruct similar data efficiently.
Anomalies (unusual patterns) result in higher reconstruction errors (MSE), making them identifiable.'''

fig.text(0.6, 0.1, model_explanation, fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

# Save the plot as a PNG file
plt.savefig('Anomaly_Detection_Using_Autoencoder.png')

plt.show()
