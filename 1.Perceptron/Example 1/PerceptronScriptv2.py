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

data = pd.read_csv(Dedomena2)
dataValues = pd.read_csv(Dedomena2Values)

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
predictions_test = p.predict(x_test)
x1,x2 = np.amin(x_test[:,0]),np.amax(x_test[:,0])
y1,y2 = p.calculateLine(x1),p.calculateLine(x2)
firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])
for i in range(len(x_test)):
    x,y = x_test[i]
    pY = p.calculateLine(x)
    if (y>pY):
        secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)
    else:
        firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)

fig4 = plt.figure()
fig4 = plt.title("Αποτελέσματα test μετά εκπαίδευσης")
fig4 = plt.plot(firstClassX,firstClassY,"r.",label="Κλάση 1")
fig4 = plt.plot(secondClassX,secondClassY,"bo",label="Κλάση 2")
fig4 = plt.plot((x1,x2),(y1,y2),"k-")

plt.legend()
plt.xlim([0,1])
plt.ylim([0,1])

showedARanger,showedBRanger,showedA,showedB = np.array([]),np.array([]),np.array([]),np.array([])
isARanger,isBRanger,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])

for i in range(len(x_test)):
    temp = predictions_test[i]
    tempReal = y_test[i]
    
    if (tempReal==0):
        isARanger,isA = np.append(isARanger,i),np.append(isA,tempReal)
        showedARanger,showedA = np.append(showedARanger,i),np.append(showedA,temp)
    else:
        isBRanger,isB = np.append(isBRanger,i),np.append(isB,tempReal)
        showedBRanger,showedB = np.append(showedBRanger,i),np.append(showedB,temp)

fig5 = plt.figure()
fig5 = plt.title("Στόχοι και προβλέψεις για testset")
fig5 = plt.plot(showedARanger,showedA,"mx",label="Πρόβλεψε 0")
fig5 = plt.plot(showedBRanger,showedB,"gx",label="Πρόβλεψε 1")
fig5 = plt.plot(isARanger,isA,"mo",label="Είναι 0",MarkerFaceColor='none')
fig5 = plt.plot(isBRanger,isB,"go",label="Είναι 1",MarkerFaceColor='none')
plt.legend()


'''firstClassX,firstClassY = np.array([]),np.array([])
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
p.fit(x_test,y_test,3)'''