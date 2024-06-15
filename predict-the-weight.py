import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('weight-height.csv')

# Explore the dataset
print("Dataset head:\n", df.head())

# Split the data into features (X) and target (y)
X = df[['Height (cm)']]
y = df['Weight (kg)']

# plot Height vs. Weight
plt.scatter(X, y, color='blue', label='Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight')
plt.show()

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# The intercept and coefficients
theta0 = model.intercept_
# round off to 2 decimal places
theta0 = round(theta0, 2)
print(f"Intercept (theta0) : {theta0}")

sample_data = [[180]]
prediction = model.predict(sample_data)
# round off to 2 decimal places
prediction = round(prediction[0], 2)
print(f"Prediction for sample data {sample_data[0]}: {prediction}")

# make predictions
y_pred = model.predict(X)
df['Predicted Weight (kg)'] = y_pred.round(2)
# print(df)


# Compare actual and predicted values
comparison = pd.DataFrame({
    'Actual Weight in Y ': y,
    'Predicted Weight from model ': y_pred.round(2),
    'Difference': (y - y_pred).round(2)
})
print(comparison)
