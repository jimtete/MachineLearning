import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)

##Δημιουργία MLP
nn = MLPClassifier(verbose=True,learning_rate_init=0.04,max_iter=100,
                   activation='tanh', solver = 'sgd',hidden_layer_sizes=(128))




##Εκπαίδευση MLP
model = nn.fit(x_train,y_train.values.ravel())

##Ποσοστό επιτυχίας στο Train

predTrain = nn.predict(x_train)

a=y_train.values.ravel()

#count = 0
#for i in range(len(predTrain)):
    #if predTrain[i]==a[i]:
       # count=count+1


#print("Accuracy (train): "+str(100*count/len(predTrain))+"%")


##Ποσοστό επιτυχίας στο Τέστ

predTest = nn.predict(x_test)

b=y_test.values.ravel()

#count = 0
#for i in range(len(predTest)):
    #if predTest[i]==b[i]:
        #count=count+1


#print("Accuracy (test): "+str(100*count/len(predTest))+"%")



##3d απεικόνιση


x0,y0,z0 = np.array([]),np.array([]),np.array([])
x1,y1,z1 = np.array([]),np.array([]),np.array([])

d1,d2 = x_test.to_numpy(),y_test.to_numpy()

for s in range(len(predTest)):
    
    x = x_test.iloc[s,0]
    y = x_test.iloc[s,1]
    z = x_test.iloc[s,2]
    
    #print(d1[s,:],' :gay: ',predTest[s])
    #print(s," is : ",predTest[s])
    
    
    
    
    
    if (predTest[s]<0.5):
        #print(x,y,z," is red")
        x0 = np.append(x0,x)
        y0 = np.append(y0,y)
        z0 = np.append(z0,z)
    elif (predTest[s]>0.5):
        #print(x,y,z," is blue")
        x1,y1,z1 = np.append(x1,x),np.append(y1,y),np.append(z1,z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x0, y0, z0, c='red', marker='x')
ax.scatter(x1, y1, z1, c='blue', marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

    