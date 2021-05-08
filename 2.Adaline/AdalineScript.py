import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from Adaline import Adaline
import pdr


# Εισαγωγή δεδομένων

df = pd.read_csv('./iris.data',header=None)


data = df.iloc[:,[0,2]] ##0,2 για την πρώτη και την τρίτη στήλη
dataValues = df.iloc[:,4]

x_train,x_test,y_train,y_test = train_test_split(data,dataValues, test_size=0.2)
x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

##Πρώτο γράφημα

firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])
thirdClassX,thirdClassY = np.array([]),np.array([])

for i in range(len(x_train)):
    x = x_train[i,0]
    y = x_train[i,1]
    
    if (y_train[i]=='Iris-setosa'):
        firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)
    elif (y_train[i]=='Iris-virginica'):
        secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)
    else:
        thirdClassX,thirdClassY = np.append(thirdClassX,x),np.append(thirdClassY,y)

   
# Γράφημα 1
#pdr.plot_decision_regions(X_std, y, classifier = model1)
fig = plt.figure()

fig = plt.plot(firstClassX,firstClassY,"r+",label="Iris-setosa")
fig = plt.plot(secondClassX,secondClassY,"bo",label="Iris-versicolor")
fig = plt.plot(thirdClassX, thirdClassY, "gx",label="Iris-virginica")

fig = plt.legend()

# Δημιουργούμε ένα δίκτυο Adaline
maxEpochs=100

model1 = Adaline(lr=0.001,epochs=maxEpochs)

# Εκπαιδεύουμε το μοντέλο
model1.fit(x_train, y_train)
predictions = model1.predict(x_train)

