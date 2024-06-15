# StandardScaler

import numpy as np
from sklearn.preprocessing import StandardScaler

# sample data with different scales
data = np.array([
    [25, 50000],
    [35, 60000],
    [45, 80000],
    [20, 45000],
    [50, 90000],
])

# create the scaler
scaler = StandardScaler()

# fit and transform the data
scaled_data = scaler.fit_transform(data)

# print original data and the scaled data
print("\nOriginal Data:\n", data)
print("\nScaled Data:\n", scaled_data)

