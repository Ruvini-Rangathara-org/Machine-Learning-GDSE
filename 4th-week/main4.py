# # SVM Binary Classification
#
# # Problem Statement
#
# # The Iris dataset contains 150 samples from three species of
# # Iris flowers setosa versicolor and virginica.
# # Each sample includes four features: sepal length,
# # sepal width, petal length, and petal width. For this example, we will
# #
# # 1. Consider only ten zanessa and 'venasto
# # 2. Use only two features seallergy and sepal net


from sklearn import datasets
import numpy as np
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

data = [[4.9, 3.0]]
prediction = clf.predict(data)
print(f"\nPrediction for sample data {data}: {iris.target_names[prediction][0]}")

import matplotlib.pyplot as plt


# Define a function to visualize the decision boundary
def plot_decision_boundary(clf, X, y):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundary')
    plt.show()


# Plot the decision boundary
plot_decision_boundary(clf, X, y)

# ////////////////////////////////////////////////////////////////////////////


# # Compare the support vector machine (SVM) with the Decision Tree Classifier
#
# # Tran a Decision tree classifier on the same dataset
# # Evaluate ite performance
# # Visualize its decision boundary
# # Compare the results with the SVM Classifier


# Train the Decision Tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Evaluate the Decision Tree model
y_pred_tree = tree_clf.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"\nDecision Tree Accuracy: {accuracy_tree:.2f}")

# Predict with Decision Tree model
prediction_tree = tree_clf.predict(data)
print(f"\nDecision Tree Prediction for sample data {data}: {iris.target_names[prediction_tree][0]}")


# Define a function to visualize the decision boundary
def plot_decision_boundary(clf, X, y, title):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(title)
    plt.show()


# Plot the decision boundaries
plot_decision_boundary(clf, X, y, 'SVM Decision Boundary')
plot_decision_boundary(tree_clf, X, y, 'Decision Tree Decision Boundary')
