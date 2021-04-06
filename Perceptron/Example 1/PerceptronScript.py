# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:54:12 2021

@author: Dimitris
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

"""Creating the data"""
n=100
numbers = np.zeros(shape=(n,2))
classes = np.zeros(shape=(n))


for i in range(n):
    key =  (random.randint(0, 1))
    if (key==1):
        numbers[i] = [(random.uniform(0.7,0.9)),(random.uniform(0.7,0.9))]
    else:
        numbers[i] = [(random.uniform(0.0,0.3)),(random.uniform(0.0,0.3))]
    classes[i] = key
    
"""End creation of data"""


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X_train, X_test, y_train, y_test = numbers[0:80],numbers[81:-1],classes[0:80],classes[81:-1]

p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0], X_train[:,1],marker='o',c=y_train)

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])
ax.set_ylim([0,1])

plt.show()