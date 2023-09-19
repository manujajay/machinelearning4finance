# Use of supervised algorithms to predict the likelihood of a borrower defaulting on a loan.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample data: [income, age, loan_amount]
# The target variable is 'creditworthy', where 1 means creditworthy and 0 means not creditworthy
data = {
    'income': [50000, 75000, 30000, 100000, 65000, 42000, 120000, 110000, 95000, 67000],
    'age': [25, 45, 35, 50, 23, 33, 55, 40, 48, 20],
    'loan_amount': [25000, 50000, 15000, 100000, 45000, 27000, 80000, 38000, 62000, 20000],
    'creditworthy': [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Separate the features (X) from the target variable (y)
X = df[['income', 'age', 'loan_amount']]
y = df['creditworthy']

# Plotting data points
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['income'], df['creditworthy'], c=df['creditworthy'])
plt.xlabel('Income')
plt.ylabel('Creditworthy')
plt.title('Income vs Creditworthiness')

plt.subplot(1, 3, 2)
plt.scatter(df['age'], df['creditworthy'], c=df['creditworthy'])
plt.xlabel('Age')
plt.ylabel('Creditworthy')
plt.title('Age vs Creditworthiness')

plt.subplot(1, 3, 3)
plt.scatter(df['loan_amount'], df['creditworthy'], c=df['creditworthy'])
plt.xlabel('Loan Amount')
plt.ylabel('Creditworthy')
plt.title('Loan Amount vs Creditworthiness')

plt.tight_layout()
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')

# Function to predict if an individual is creditworthy
def predict_creditworthiness(income, age, loan_amount):
    prediction = clf.predict([[income, age, loan_amount]])[0]
    
    if prediction == 1:
        return "The individual is creditworthy."
    else:
        return "The individual is not creditworthy."

# Example usage of the prediction function
print(predict_creditworthiness(70000, 30, 40000))  # Should generally return "The individual is creditworthy."
print(predict_creditworthiness(30000, 25, 60000))  # Should generally return "The individual is not creditworthy."
