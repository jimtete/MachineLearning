import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

##Εισαγωγή δεδομένων

Dedomena1 = "./exported_data_pack_3D1.csv"
Dedomena2 = "./exported_data_pack_3D2.csv"
Dedomena1Values = "./exported_data_pack_values_3D1.csv"
Dedomena2Values = "./exported_data_pack_values_3D2.csv"


##Διαχωρισμός δεδομένων σε τιμές
data = pd.read_csv(Dedomena1)
dataValues = pd.read_csv(Dedomena1Values)

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2, random_state=4)

##Δημιουργία MLP
nn = MLPClassifier(verbose=True,learning_rate_init=0.004,max_iter=100,activation='tanh', solver = 'sgd',hidden_layer_sizes=(128), random_state=1)


##Εκπαίδευση MLP
model = nn.fit(x_train,y_train.values.ravel())

print(model)

##Ποσοστό επιτυχίας στο Train

predTrain = nn.predict(x_train)

a=y_train.values.ravel()

count = 0
for i in range(len(predTrain)):
    if predTrain[i]==a[i]:
        count=count+1


print("Accuracy (train): "+str(100*count/len(predTrain))+"%")


##Ποσοστό επιτυχίας στο Τέστ

predTest = nn.predict(x_test)

b=y_test.values.ravel()

count = 0
for i in range(len(predTest)):
    if predTest[i]==b[i]:
        count=count+1


print("Accuracy (test): "+str(100*count/len(predTest))+"%")



##3d απεικόνιση
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for s in range(len(predTrain)):
    
    x = x_train.iloc[:,0]
    y = x_train.iloc[:,1]
    z = x_train.iloc[:,2]
    
    
    if predTrain[s]==0:
        ax.scatter(x, y, z, c='red', marker='x')
    if predTrain[s]==1:
        ax.scatter(x, y, z, c='blue', marker='x')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.show()
    