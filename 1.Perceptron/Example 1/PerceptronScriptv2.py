import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from perceptron import Perceptron
import pandas as pd

def calculateLine(x,p):   
    return (-p.weights[0]*x-p.bias) / p.weights[1]


##Εισαγωγή Δεδομένων
Dedomena1, Dedomena1Values = "./exported_data_pack_1.csv","./exported_data_pack_values_1.csv"
Dedomena2, Dedomena2Values = "./exported_data_pack_2.csv","./exported_data_pack_values_2.csv"
Dedomena3, Dedomena3Values = "./exported_data_pack_3.csv","./exported_data_pack_values_3.csv"
Dedomena4, Dedomena4Values = "./exported_data_pack_4.csv","./exported_data_pack_values_4.csv"

##Διαχωρισμός των δεδομένων σε τιμές 

data = pd.read_csv(Dedomena1)
dataValues = pd.read_csv(Dedomena1Values)

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)

x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
##Εκκίνηση διαδικασίας εκπαίδευσης
epochs=69

##Πρώτο γράφημα

firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])

for i in range(len(x_train)):
    x = x_train[i,0]
    y = x_train[i,1]
    
    if (y_train[i]<0.5):
        firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)
    else:
        secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)

fig = plt.figure()

fig = plt.plot(firstClassX,firstClassY,"r.",label="Κλάση 1")
fig = plt.plot(secondClassX,secondClassY,"bo",label="Κλάση 2")

fig = plt.legend()





##Δεύτερο γράφημα // διαδικασία εκπαίδευσης

#for index in range(epochs):
p = Perceptron(learning_rate=0.1, n_iters=epochs)
p.fit(x_train, y_train,2)
predictions = p.predict(x_train)
        
##Τρίτο γράφημα
p.fit(x_train,y_train,3)
  
##Τέταρτα γράφηματα

firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])

for i in range(len(x_test)):
    x = x_test[i,0]
    y = x_test[i,1]
    
    if (y_test[i]<0.5):
        firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)
    else:
        secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)

fig2 = plt.figure()

fig2 = plt.plot(firstClassX,firstClassY,"r.",label="Κλάση 1")
fig2 = plt.plot(secondClassX,secondClassY,"bo",label="Κλάση 2")

fig2 = plt.legend()

p.fit(x_test,y_test,2)
p.fit(x_test,y_test,3)