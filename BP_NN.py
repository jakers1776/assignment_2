#Backprop template network

import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
# Load the dataset
# Load the dataset
data = load_iris()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size = 0.2, random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

#Neural Network

# Initialize neural network object and fit object - attempt 1

lr_nn_model1 = MLPClassifier(learning_rate="constant", learning_rate_init=0.001,max_iter=2000,random_state=3)

lr_nn_model1.fit(X_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = lr_nn_model1.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print('Train accuracy: ',y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = lr_nn_model1.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print('Test accuracy: ',y_test_accuracy)
