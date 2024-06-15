import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Read the CSV file
df = pd.read_csv('music.csv')

# Display the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Display the first few rows of the DataFrame
print("DataFrame:\n", df.head())

# Prepare the data
x = df.drop(columns=['genre'])  # Features
y = df['genre']  # Target variable

# Initialize the model
model = DecisionTreeClassifier()

# Train the model
model.fit(x, y)

# Make predictions
predictions = model.predict([[21, 1]])

# Display predictions
print("Predictions for [21, 1]:", predictions)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
predictions2 = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_test, predictions2)

# Display accuracy
print("Accuracy Score:", score)


joblib.dump(model, 'music-recommender.joblib')

model = joblib.load('music-recommender.joblib')

predictions = model.predict([[21, 1], [22, 0]])
