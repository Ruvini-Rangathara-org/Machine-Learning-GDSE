import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
data = {
    'Size (sqft)': [2104, 1600, 2400, 1416, 3000],
    'Bedrooms': [3, 3, 3, 2, 4],
    'Price ($)': [399900, 329900, 369000, 232000, 539900]
}

# Create DataFrame
df = pd.DataFrame(data)

# Explore the dataset
print("Dataset head:\n", df.head())

# Split the data into features (X) and target (y)
X = df[['Size (sqft)', 'Bedrooms']]
y = df['Price ($)']


# plot Size vs. Price
plt.scatter(df['Size (sqft)'], df['Price ($)'], color='blue', label='Size')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Size vs Price')
plt.show()

# plot Bedrooms vs. Price
plt.scatter(df['Bedrooms'], df['Price ($)'], color='green', label='Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Price ($)')
plt.title('Bedrooms vs Price')
plt.show()


# Train a linear regression model
model = LinearRegression()
model.fit(X, y)


# The intercept and coefficients
theta0 = model.intercept_
print(f"Intercept (theta0) : {theta0}")

theta_1, theta_2 = model.coef_
print(f"Coefficient for size (theta_1) : {theta_1}")
print(f"Coefficient for bedrooms (theta_2) : {theta_2}")

print('\n\n')

# Predict the price from sample data
sample_data = [[2500, 3]]
prediction = model.predict(sample_data)
print(f"Prediction for sample data {sample_data[0]}: {prediction[0]:.2f}")


# Make predictions
y_pred = model.predict(X)
# print as a table
df['Predicted Price ($)'] = y_pred.round(2)
print(df)

print('\n\n')

# Compare actual and predicted values
comparison = pd.DataFrame({
    'Actual Price in Y ': y,
    'Predicted Price from model ': y_pred.round(2),
    'Difference': (y - y_pred).round(2)
})
print(comparison)




