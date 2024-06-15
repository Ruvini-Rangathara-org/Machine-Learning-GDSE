import pandas as pd

# Read the CSV file
df = pd.read_csv('loan-data.csv')

# Display the first few rows of the DataFrame
print("\n\nDataFrame:\n\n", df.head())

# Display the summary statistics of the DataFrame
print("\n\nSummary Statistics of the DataFrame:\n\n", df.describe())


# Check for the missing values
print("\n\nMissing values:\n\n", df.isnull().sum())

# feature and target variable separation
x = df[['age', 'income', 'loan_amount', 'credit_score']]
y = df['default']

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# import warning
import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LogisticRegression

# initialize the model
model = LogisticRegression()

# train the model
model.fit(x_train, y_train)

LogisticRegression()


# make predictions
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score
prediction = precision_score(y_test, y_pred)

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)


print("\n\nAccuracy Score: ", accuracy_score)
print("Precision Score: ", prediction)
print("Recall Score: ", recall)
print("Confusion Matrix: ", confusion)

# example : new customer data
new = [[30, 50000, 10000, 650]]
prediction = model.predict(new)
print("\n\nPrediction for new customer: ", prediction)
