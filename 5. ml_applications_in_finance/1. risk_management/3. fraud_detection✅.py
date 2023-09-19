# Anomaly detection to identify unusual patterns which could suggest fraudulent transactions.

'''
a Python script that demonstrates a simple approach to fraud detection using machine learning.
In this example, we'll use the RandomForestClassifier from scikit-learn to classify transactions as either "fraudulent" or "genuine".
'''

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for demonstration
# In a real-world application, replace this with actual data
np.random.seed(42)
n_samples = 1000

# Genuine transactions are centered around (0, 0)
genuine = np.random.normal(0, 1, (int(n_samples * 0.95), 2))
genuine_labels = np.zeros(int(n_samples * 0.95))

# Fraudulent transactions are centered around (5, 5)
fraud = np.random.normal(5, 1, (int(n_samples * 0.05), 2))
fraud_labels = np.ones(int(n_samples * 0.05))

# Combine into one dataset
X = np.vstack([genuine, fraud])
y = np.hstack([genuine_labels, fraud_labels])

# Data Preprocessing: Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train RandomForest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluation Metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

# Plotting
plt.figure(figsize=(10, 6))

# Plot genuine transactions
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label='Genuine', alpha=0.5)

# Plot fraudulent transactions
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='Fraud', alpha=0.5)

# Highlight false negatives
plt.scatter(X_test[(y_test == 1) & (y_pred == 0)][:, 0], X_test[(y_test == 1) & (y_pred == 0)][:, 1], s=100,
            facecolors='none', edgecolors='r', label='False Negative')

# Highlight false positives
plt.scatter(X_test[(y_test == 0) & (y_pred == 1)][:, 0], X_test[(y_test == 0) & (y_pred == 1)][:, 1], s=100,
            facecolors='none', edgecolors='m', label='False Positive')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Fraud Detection')
plt.legend()
plt.show()

'''
Explanation:

Data Generation: We're creating synthetic data for both genuine and fraudulent transactions. In practice, you would replace this with your actual data.

Feature Scaling: Using the StandardScaler from scikit-learn to normalize features, which is often necessary for machine learning algorithms.

Train-Test Split: We're splitting the data into training and test sets, with 20% of the data reserved for testing.

Random Forest Classifier: A simple Random Forest model is trained on the training data.

Prediction and Evaluation: We then use the trained model to make predictions on the test set, and print evaluation metrics like accuracy, confusion matrix, and classification report.

Plotting: Finally, we plot the test data, highlighting genuine and fraudulent transactions. We also indicate false positives and false negatives.

This is a simplified example meant for demonstration. Real-world fraud detection models would involve far more complexity, such as dealing with imbalanced data, feature engineering, hyperparameter tuning, and possibly using more advanced algorithms.
'''