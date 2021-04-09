# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:42:51 2021

@author: Dimitris
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from perceptron import Perceptron
import pandas as pd

"""Loading the data"""

##You can only change the value below, number:1 indicates the first datapack and so on
datapack = pd.read_csv('./exported_data_pack_1.csv')
##End of change


numbers = datapack.to_numpy()
numbers_train = numbers[:180]
numbers_test = numbers[181:200]

##You can only change the value below, number:1 indicates the first datapack and so on
datapack_values = pd.read_csv('./exported_data_pack_values_1.csv')
##End of change



classes = datapack_values.to_numpy()
classes_train = classes[:180]
classes_test = classes[181:200]

    
"""End loading the data"""


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X_train, X_test, y_train, y_test = numbers_train,numbers_test,classes_train,classes_test

p = Perceptron(learning_rate=0.1, n_iters=100)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))



ax = plt.axes(projection='3d')

z = np.linspace(0,30,100)
x = np.sin(z)
y = np.cos(z)

ax.plot3D(x,y,z)

plt.show()