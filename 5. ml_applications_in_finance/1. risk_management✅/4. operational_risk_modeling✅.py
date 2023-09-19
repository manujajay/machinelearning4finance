# Predictive models to identify potential operational hazards and risks.

'''
a Python script that demonstrates how to use machine learning to predict operational hazards based on synthetic features.

In this example, I'll use a RandomForestClassifier from scikit-learn to create a predictive model.
The target variable is a binary outcome representing whether or not an operational hazard exists (1 for hazard, 0 for no hazard).
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
# In a real-world application, you would use actual data
np.random.seed(42)
n_samples = 1000

# Non-hazardous situations with features centered around (2, 2)
non_hazard = np.random.normal(2, 1, (int(n_samples * 0.7), 2))
non_hazard_labels = np.zeros(int(n_samples * 0.7))

# Hazardous situations with features centered around (5, 5)
hazard = np.random.normal(5, 1, (int(n_samples * 0.3), 2))
hazard_labels = np.ones(int(n_samples * 0.3))

# Combine into one dataset
X = np.vstack([non_hazard, hazard])
y = np.hstack([non_hazard_labels, hazard_labels])

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluation Metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot non-hazardous situations
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label='Non-Hazard', alpha=0.6)

# Plot hazardous situations
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='Hazard', alpha=0.6)

# Highlight false negatives
plt.scatter(X_test[(y_test == 1) & (y_pred == 0)][:, 0], X_test[(y_test == 1) & (y_pred == 0)][:, 1], s=100,
            facecolors='none', edgecolors='r', label='False Negative')

# Highlight false positives
plt.scatter(X_test[(y_test == 0) & (y_pred == 1)][:, 0], X_test[(y_test == 0) & (y_pred == 1)][:, 1], s=100,
            facecolors='none', edgecolors='m', label='False Positive')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Operational Hazard Prediction')
plt.legend()
plt.show()

'''
Explanation:

Data Generation: Synthetic data is generated to simulate operational hazards. In practice, you would use actual operational data with relevant features.

Feature Scaling: Features are scaled using StandardScaler from scikit-learn. This is often necessary for machine learning algorithms.

Train-Test Split: The dataset is split into a training set and a test set.

Random Forest Classifier: A RandomForestClassifier is trained on the training set.

Prediction and Evaluation: The model is used to make predictions on the test set, and several evaluation metrics are printed.

Plotting: A scatter plot of the test set is created where true positives, true negatives, false positives, and false negatives are marked. This provides a visual insight into how well the model is performing.

This is a simplified example intended for demonstration purposes. A real-world application would include more steps like feature engineering, dealing with imbalanced data, hyperparameter tuning, and perhaps the use of more advanced machine learning algorithms.
'''