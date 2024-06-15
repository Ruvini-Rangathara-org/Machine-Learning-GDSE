# Predicting Student Performance

# we want to predict whether student will pass or will fail or fail based on their attendance with SVM & Decision Tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
df = pd.read_csv('student_performance_large.csv')

# Display the first few rows of the dataset to understand its structure
# print(df.head())

# Step 3: Separate features (X) and target variable (y)
X = df.drop('Pass/Fail', axis=1)  # Features: all columns except 'Pass/Fail'
y = df['Pass/Fail']               # Target variable: 'Pass/Fail' column

# Get feature names
feature_names = X.columns

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Decision Tree classifier
dt_clf = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier
dt_clf.fit(X_train, y_train)

# Step 6: Make predictions using Decision Tree classifier
dt_y_pred = dt_clf.predict(X_test)

# Calculate accuracy of Decision Tree classifier
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f' Decision Tree Accuracy: {dt_accuracy:.2f}')

# Display classification report for Decision Tree classifier
# print("Decision Tree Classification Report:")
# print(classification_report(y_test, dt_y_pred))



# Step 7: Build the SVM classifier
svm_clf = SVC(random_state=42)

# Train the SVM classifier
svm_clf.fit(X_train, y_train)

# Step 8: Make predictions using SVM classifier
svm_y_pred = svm_clf.predict(X_test)

# Calculate accuracy of SVM classifier
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f' SVM Accuracy: {svm_accuracy:.2f}')

# Display classification report for SVM classifier
# print("SVM Classification Report:")
# print(classification_report(y_test, svm_y_pred))


# sample data
sample_data = [[8, 87]]
prediction = svm_clf.predict(sample_data)
print(f"\nPrediction for sample data {sample_data}: {prediction[0]}")

# sample data
sample_data = [[8, 87]]
prediction = dt_clf.predict(sample_data)
print(f"\nPrediction for sample data {sample_data}: {prediction[0]}")