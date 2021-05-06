import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

Dedomena1 = "./exported_data_pack_3D1.csv"
Dedomena2 = "./exported_data_pack_3D2.csv"
Dedomena1Values = "./exported_data_pack_values_3D1.csv"
Dedomena2Values = "./exported_data_pack_values_3D2.csv"

data = pd.read_csv(Dedomena1)
dataValues = pd.read_csv(Dedomena1Values)

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)

nn = MLPClassifier(verbose=True,learning_rate_init=0.04,max_iter=1000,activation='tanh', solver = 'sgd',hidden_layer_sizes=(128))

model = nn.fit(x_train,y_train.values.ravel())

