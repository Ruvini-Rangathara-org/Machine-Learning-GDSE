# Import necessary libraries
from sklearn.datasets import load_iris  # For loading the iris dataset
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.tree import DecisionTreeClassifier  # For the decision tree classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For evaluating the model
import pandas as pd  # For data manipulation and analysis

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features (inputs)
y = iris.target  # Target (outputs)

# Convert to DataFrame for easier exploration
df = pd.DataFrame(X, columns=iris.feature_names)  # Create DataFrame with feature names as columns
df['species'] = y  # Add target column to DataFrame

df.head()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)  # Initialize the classifier
clf.fit(X_train, y_train)  # Fit the classifier to the training data

# Make predictions
y_pred = clf.predict(X_test)  # Predict on the test data

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Display classification report

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))  # Display confusion matrix

# Test with a sample data point
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Define sample data
prediction = clf.predict(sample_data)  # Predict using the classifier
print(f"\nPrediction for sample data {sample_data}: {iris.target_names[prediction][0]}")  # Display prediction
