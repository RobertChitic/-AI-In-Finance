# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classification (for classification tasks)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Load and prepare the data
df = pd.read_csv('stock_data.csv')  # Replace with your file path

# Debug: Inspect column names
print("Original columns:", df.columns)

# Clean column names (strip whitespace and make lowercase for consistency)
df.columns = df.columns.str.strip().str.lower()
print("Cleaned columns:", df.columns)

# Check if 'value' exists and adjust accordingly
if 'value' not in df.columns:
    raise KeyError("The column 'value' was not found in the dataset. Check the column names.")

# Step 3: Feature Engineering
df['price_change'] = df['value'].shift(-1) - df['value']  # Calculate price change
df['target'] = (df['price_change'] > 0).astype(int)  # 1 if price went up, 0 if price went down

# Drop rows with missing values (last row after shift)
df = df.dropna()

# Step 4: Define features and target
df['previous_value'] = df['value'].shift(1)  # Previous day's value
df = df.dropna()  # Drop missing values created by the shift

X = df[['previous_value']]  # Feature: Previous day's value
y = df['target']  # Target: 1 (up) or 0 (down)

# Step 5: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

# Step 7: Initialize the SVM model
svm_model = SVC(kernel='linear')  # Linear kernel for simplicity

# Step 8: Train the SVM model
svm_model.fit(X_train_scaled, y_train)

# Step 9: Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Visualize Results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', linestyle='-', marker='o', color='blue')
plt.plot(y_pred, label='Predicted', linestyle='-', marker='x', color='red')
plt.title('Stock Price Prediction: Actual vs Predicted')
plt.xlabel('Test Samples')
plt.ylabel('Stock Price Direction (0 = Down, 1 = Up)')
plt.legend()
plt.show()
