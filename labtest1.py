# NAME - Mantej Singh Sohi
# ROLL NO - 22AG10024

# Important modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Implementation of closed form linear regression
class linear_regression_closed:
  def __init__(self):
    pass

  def fit(self, X, y):
    self.X_train= X
    self.y_train= y

    # adding an extra column containg 1's to the self.X_train to take care of bias terms
    self.X_train = np.column_stack((np.ones(np.shape((self.X_train))[0]), self.X_train))

    # Calculate the parameters using the closed-form solution
    theta = (np.linalg.inv(self.X_train.T @ self.X_train)) @ self.X_train.T @ self.y_train
    return theta

  def predict(self, X_test, theta):
    # adding an extra column containg 1's to the self.X_test to take care of bias terms
    X_test = np.column_stack((np.ones(np.shape(X_test)[0]), X_test))
    predictions= np.dot(X_test, theta)
    for i in range (predictions.shape[0]):
      if(predictions[i]>=0.5):
        predictions[i] = 1
      else:
        predictions[i] = 0
    return predictions

  def accuracy(self, y, predictions):
    # Fxn calculating accuracy
    accu = (predictions == y)
    accuracy= (np.sum(accu))/(accu.shape[0])
    return accuracy*100

# Implementation of gradient descent linear regression
class linear_regression_gradient:
  def __init__(self, iters, alpha):
    self.iters= iters
    self.alpha= alpha

  def fit(self, X, y):
    weights, bias= self._fit(X, y)
    return weights, bias

  def _fit(self, X, y):
    weights= np.zeros(((X.shape[1]), 1))
    bias= 0

    for i in range (self.iters):
      dj_dw, dj_db= self.compute_gradient(X, y, weights, bias)
      weights= weights - ((dj_dw)*self.alpha)
      bias= bias - ((dj_db)*self.alpha)

    return weights, bias

  def compute_gradient(self, X, y, w, b):
    m = X.shape[0]
    predictions= np.dot(X, w)+ b
    errors= predictions- y
    dj_dw= (X.T.dot(errors))/m
    dj_db= (np.sum(errors))/m

    return dj_dw, dj_db

  def predict(self, x, weights, bias):
    predictions= (np.dot(x,weights) + bias)

    for i in range (predictions.shape[0]):
      if(predictions[i]>=0.5):
        predictions[i] = 1
      else:
        predictions[i] = 0

    return predictions

  def accuracy(self, y, predictions):

    # Fxn calculating accuracy
    accu = (predictions == y)
    accuracy= (np.sum(accu))/(accu.shape[0])
    return accuracy*100

# Experiment 1
df= pd.read_csv("diabetes.csv")
un_nec_columns= ["Pregnancies", "SkinThickness", "DiabetesPedigreeFunction"]
dataset_altered= df.drop(columns= un_nec_columns, axis= 0, inplace=False)

# Shuffling the dataset_altered
dataset_altered = dataset_altered.sample(frac=1).reset_index(drop=True)

# Displaying first ten rows of dataset_altered
dataset_altered.head(10)

# Experiment 2
# Calculate correlation coefficients (targets included)
correlation_matrix = dataset_altered.corr()
print(correlation_matrix)
print("\n")

# Creating a heatmap for the calculated correlations using seaborn
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
print("\n")

# Report
print('''REPORT:-

On, a general note, I am not able to find any negative co-relations (except one). This signifies that any pair of variables I take, they
are positively co-related to each other which gives a feeling that linear regression can do well on this problem.

But, the +ve co-relations are not too strong as well. (i.e. co-relation values in range (0.5, 1) are null.

Glucose seems to be the most co-related parameter with the outcome and BP is least co-related. Co-relation value of BP and outcome is ~ 0.065
which is very close to zero and suggests that they are near to independent. From this we can make an inference that weight associated with
Bp should be small and weight associated to Glucose should be significant.
''')

# Experiment 3
# Extracting out features and target dataframes explicitly from dataset_altered
features= dataset_altered.iloc[:, :5]
targets= dataset_altered.iloc[:, 5]

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=2347)
# print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

# Running closed form linear regression on training and testing data ; claculated accuracy of the model afterwards
model= linear_regression_closed()
parameters= model.fit(X_train, y_train)
predictions_train = model.predict(X_train, parameters)
predictions_test = model.predict(X_test, parameters)

accuracy_train = model.accuracy(y_train, predictions_train)
accuracy_test = model.accuracy(y_test, predictions_test)

# print(parameters)
# print(predictions_train)
# print(predictions_test)
# print(accuracy_train)
# print(accuracy_test)

print(f"Accuracy achieved after running closed form on training data = {accuracy_train}")
print(f"Accuracy achieved after running closed form on test data = {accuracy_test}")

# Plotting Confusion matrices
conf_matrix = confusion_matrix(y_train, predictions_train)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels= [0,1], yticklabels= [0,1])
plt.title('Confusion Matrix for Training set')
plt.xlabel('Predicted Outcome')
plt.ylabel('True Outcome')
plt.show()

conf_matrix = confusion_matrix(y_test, predictions_test)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="OrRd", xticklabels= [0,1], yticklabels= [0,1])
plt.title('Confusion Matrix for Testing set')
plt.xlabel('Predicted Outcome')
plt.ylabel('True Outcome')
plt.show()

# Experiment 4

# Extracting out features and target dataframes explicitly from dataset_altered
features= dataset_altered.iloc[:, :5]
targets= dataset_altered.iloc[:, 5]

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=2347)

# Running gradient descent linear regression on testing data ; for different values of learning rate
alpha= [0.00001,0.001,0.05,0.1]
accuracy= []

y_train= np.array(y_train)
y_test= np.array(y_test)
y_train= y_train.flatten()
y_test= y_test.flatten()

for i in alpha:
  model= linear_regression_gradient(50, i)
  weights, bias= model.fit(X_train, y_train)
  predictions = model.predict(X_test, weights, bias)
  _accuracy = model.accuracy(y_test, predictions)
  accuracy.append(_accuracy)

print(accuracy)