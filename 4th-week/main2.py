import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
iris = load_iris()
X = iris.data  # Features (inputs)
y = iris.target  # Target (outputs)

# Convert to DataFrame for exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Explore the dataset
print("\n\nDataset Head:\n", df.head())
print("\n\nDataset Description:\n", df.describe())
print("\n\nDataset Info:\n", df.info())
print("\n\nClass Distribution:\n", df['target'].value_counts())

# Preprocess the data
# For this dataset, no preprocessing is required as it is already clean and ready for use

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Example: New flower data
new = [[5.1, 3.5, 1.4, 0.2]]
new_prediction = model.predict(new)
predicted_class = iris.target_names[new_prediction[0]]
print("\n\nPrediction for new flower: ", predicted_class)
