# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:54:12 2021

@author: Dimitris
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from perceptron3D import Perceptron
import pandas as pd

"""Loading the data"""

##You can only change the value below, number:1 indicates the first datapack and so on
datapack = pd.read_csv('./exported_data_pack_3D2.csv')
##End of change


numbers = datapack.to_numpy()
numbers_train = numbers[:180]
numbers_test = numbers[181:200]

##You can only change the value below, number:1 indicates the first datapack and so on
datapack_values = pd.read_csv('./exported_data_pack_values_3D2.csv')
##End of change



classes = datapack_values.to_numpy()
classes_train = classes[:180]
classes_test = classes[181:200]

    
"""End loading the data"""


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X_train, X_test, y_train, y_test = numbers_train,numbers_test,classes_train,classes_test


epochs=100
for index in range(epochs):
    p = Perceptron(learning_rate=0.1, n_iters=index)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plt.scatter(X_train[:,0], X_train[:,1], X_train[:,2],zdir=X_train[:,2],depthshade=True, marker='o',c=y_train)
    
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x0_1 = np.amin(X_train[:,0])
    x0_2 = np.amax(X_train[:,0])
    
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    
    
    
    ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k') 
    
    ymin = np.amin(X_train[:,1])
    ymax = np.amax(X_train[:,1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_zlim([0,1])
    
    plt.show()