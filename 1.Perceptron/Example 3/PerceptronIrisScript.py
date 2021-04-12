# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:24:08 2021

@author: Dimitris
"""

import pandas as pd

df = pd.read_csv('./iris.data',header=None)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

y = df.iloc[:,4].values

y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[:, [0,2]].values

from PerceptronIris import Perceptron

def plot_decision_regions(X, y, classifier, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot class samples
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
      alpha=0.8, c=cmap(idx),
      marker=markers[idx], label=cl)


epochs = 10
for index in range(epochs):
    pn = Perceptron(0.1,index)
    pn.fit(X,y)
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:150, 0], X[50:150, 1], color='blue', marker='x', label='versicolor or virginica')
    plot_decision_regions(X, y, classifier=pn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('number of misclassifications')
plt.show()