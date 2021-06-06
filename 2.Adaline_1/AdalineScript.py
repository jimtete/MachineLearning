import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from mlxtend.classifier import Adaline

def calculateLine(a,x):
    return (-a.w_[0]*x-a.b_) / a.w_[1]


##Εισαγωγή δεδομένων

Dedomena1, Dedomena1Values = "./exported_data_pack_1.csv","./exported_data_pack_values_1.csv"
Dedomena2, Dedomena2Values = "./exported_data_pack_2.csv","./exported_data_pack_values_2.csv"
Dedomena3, Dedomena3Values = "./exported_data_pack_3.csv","./exported_data_pack_values_3.csv"
Dedomena4, Dedomena4Values = "./exported_data_pack_4.csv","./exported_data_pack_values_4.csv"

data = pd.read_csv(Dedomena3)
dataValues = pd.read_csv(Dedomena3Values)

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)
x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

y_train,y_test = y_train.astype(type(np.int16(0).item())),y_test.astype(type(np.int16(0).item()))
y_train,y_test = y_train.flatten(),y_test.flatten()

##Γράφημα 1
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
fig = plt.title("Τα στοιχεία στις 2 διαστάσεις")
fig = plt.plot(firstClassX,firstClassY,"r.",label="Κλάση 1")
fig = plt.plot(secondClassX,secondClassY,"bo",label="Κλάση 2")

fig = plt.legend()

##Γράφημα 2
Epochs=100
learning_rate=0.001
predictions_train_array = np.ones((Epochs,len(y_train)))


for index in range(Epochs): 
    d=x_train
    dv=y_train
    
    
    a = Adaline(epochs=index,eta=learning_rate,minibatches=len(dv))
    a.fit(x_train,y_train)
    predictions = a.predict(d)
    
    ##Αποθήκευση των στόχων για χρήση αργότερα
    predictions_train_array[index,:]=predictions
    
    
    x1,x2 = np.amin(d[:,0]),np.amax(d[:,0])
    y1,y2 = calculateLine(a,x1),calculateLine(a,x2)
    firstClassX,firstClassY = np.array([]),np.array([])
    secondClassX,secondClassY = np.array([]),np.array([])

    for i in range(len(d)):
        x,y=d[i]
        if (predictions[i]==0):
            firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)
        else:
            secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)

    fig2 = plt.figure()
    fig2 = plt.title("Εκπαίδευση δικτύου adaline εποχή: "+str(index+1))
    fig2 = plt.plot(firstClassX, firstClassY, "r.",label="Κλάση 1")
    fig2 = plt.plot(secondClassX,secondClassY,"bo",label="Κλάση 2")
    fig2 = plt.plot((x1,x2),(y1,y2),"k-")
    
    plt.legend()
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    plt.show()


#Γράφημα 3

for j in range(Epochs):  
    showedARanger,showedBRanger,showedA,showedB = np.array([]),np.array([]),np.array([]),np.array([])
    isARanger,isBRanger,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    for i in range(len(d)):
        temp = predictions_train_array[j,i]
        tempReal = dv[i]
        
        if (tempReal==0):
            isARanger,isA = np.append(isARanger,i),np.append(isA,tempReal)
            showedARanger,showedA = np.append(showedARanger,i),np.append(showedA,temp)
        else:
            isBRanger,isB = np.append(isBRanger,i),np.append(isB,tempReal)
            showedBRanger,showedB = np.append(showedBRanger,i),np.append(showedB,temp)

    fig3 = plt.figure()
    fig3 = plt.title("Στόχοι και προβλέψεις για trainset, εποχή: "+str(j+1))
    fig3 = plt.plot(showedARanger,showedA,"mx",label="Πρόβλεψε 0")
    fig3 = plt.plot(showedBRanger,showedB,"gx",label="Πρόβλεψε 1")
    fig3 = plt.plot(isARanger,isA,"mo",label="Είναι 0",MarkerFaceColor='none')
    fig3 = plt.plot(isBRanger,isB,"go",label="Είναι 1",MarkerFaceColor='none')
    plt.legend()
    

#Γράφημα 4
loss = a.cost_
fig4 = plt.figure()
fig4 = plt.title("Ποσοστό σφάλματος / εποχή")
fig4 = plt.xlabel("Αριθμός εποχών")
fig4 = plt.ylabel("Ποσοστό σφάλματος")

plt.plot(np.arange(1,Epochs),loss,"b-.")
plt.legend()
plt.show()

##Γράφημα 5

d = x_test
dv = y_test
predictions = a.predict(d)

showedARanger,showedBRanger,showedA,showedB = np.array([]),np.array([]),np.array([]),np.array([])
isARanger,isBRanger,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])   
for i in range(len(d)):
        temp = predictions[i]
        tempReal = dv[i]
        
        if (tempReal==0):
            isARanger,isA = np.append(isARanger,i),np.append(isA,tempReal)
            showedARanger,showedA = np.append(showedARanger,i),np.append(showedA,temp)
        else:
            isBRanger,isB = np.append(isBRanger,i),np.append(isB,tempReal)
            showedBRanger,showedB = np.append(showedBRanger,i),np.append(showedB,temp)    
fig5 = plt.figure()
fig5 = plt.title("Στόχοι και προβλέψεις για testset, εποχή: "+str(j+1))
fig5 = plt.plot(showedARanger,showedA,"mx",label="Πρόβλεψε 0")
fig5 = plt.plot(showedBRanger,showedB,"gx",label="Πρόβλεψε 1")
fig5 = plt.plot(isARanger,isA,"mo",label="Είναι 0",MarkerFaceColor='none')
fig5 = plt.plot(isBRanger,isB,"go",label="Είναι 1",MarkerFaceColor='none')
plt.legend()