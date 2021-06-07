import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation and plotting
import matplotlib.pyplot as plt # data plotting
from LeastSquareMethod import LeastSquareMethod

from sklearn.model_selection import train_test_split

def fill_with_array(y_temp,y):
    for i in range(len(y)):
        name = y[i]
        if (name=="Iris-setosa"):
            y_temp[i]=[1,-1,-1]
        elif (name=="Iris-versicolor"):
            y_temp[i]=[-1,1,-1]
        else:
            y_temp[i]=[-1,-1,1]
    return y_temp

def fill_with_1s_and_0s(y_temp,y,key):
    for i in range(len(y)):
        name = y[i]
        if (name==key):
            y_temp[i]=1
        else:
            y_temp[i]=0
    return y_temp

df = pd.read_csv("./iris.csv")

data = df.iloc[0:,[1,3]]
dataValues = df.iloc[0:,5]

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)

x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

y_train_temp = np.zeros((120,3),dtype=int)
y_test_temp = np.zeros((30,3),dtype=int)

y_train=fill_with_array(y_train_temp,y_train)
y_test = fill_with_array(y_test_temp,y_test)



##Πρώτο γράφημα
firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])
thirdClassX,thirdClassY = np.array([]),np.array([])

for i in range(len(x_train)):
    x = x_train[i,0]
    y = x_train[i,1]
    
    if (y_train[i]==[1,-1,-1]).all():
        firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)
    elif (y_train[i]==[-1,1,-1]).all():
        secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)
    else:
        thirdClassX,thirdClassY = np.append(thirdClassX,x),np.append(thirdClassY,y)


fig = plt.figure()

fig = plt.plot(firstClassX,firstClassY,"r.",label="Iris-setosa")
fig = plt.plot(secondClassX,secondClassY,"bo",label="Iris-versicolor")
fig = plt.plot(thirdClassX,thirdClassY,"gx",label="Iris-virginica")

fig = plt.legend()


##Δεύτερο γράφημα
fig2 = plt.figure()

fig2 = plt.plot(firstClassX,firstClassY,"r.",label="[1,-1,-1]")
fig2 = plt.plot(secondClassX,secondClassY,"bo",label="[-1,1,-1]")
fig2 = plt.plot(thirdClassX,thirdClassY,"gx",label="[-1,-1,1]")

fig2 = plt.legend()

##Τρίτο γράφημα (Σύγκριση μεταξύ Iris-setosa & Iris-versicolor)
data = df.iloc[:100,[1,3]]
dataValues = df.iloc[:100,5]

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)

x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()



y_train_temp = np.zeros((80,1),dtype=int)
y_test_temp = np.zeros((20,1),dtype=int)

y_train=fill_with_1s_and_0s(y_train_temp,y_train,"Iris-setosa")
y_test = fill_with_1s_and_0s(y_test_temp,y_test,"Iris-setosa")

epochs=1
learning_rate=0.003
prediction = np.zeros(80)


weights = np.array([0,1,1])

for index in range(epochs):
    
    i=0
    for data in x_train: 
        prediction[i] = (-weights[0]+np.dot(data,np.transpose(weights[1:3])))
        print(prediction[i])       
        i+=1
    