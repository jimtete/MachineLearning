import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from Adaline import Adaline
import pandas as pd


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

y_train=y_train.reshape(160)
y_test=y_test.reshape(40)


x_temp = np.zeros((160,3))-1
x_temp[:,1:3] = x_train
x_train = x_temp#[:5]
#y_train = y_train[:5]

#x_train = np.array([[-1,0,0],
                    #[-1,0,1],
                    #[-1,1,0],
                    #[-1,1,1]])

# = np.array([-1,1,1,1])


#Εκκίνηση διαδικασίας εκπαίδευσης

Epochs = 50

##Πρώτο γράφημα

firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])

for i in range(len(x_train)):
    x = x_train[i,1]
    y = x_train[i,2]
    
    if (y_train[i]<0):
        firstClassX,firstClassY = np.append(firstClassX,x),np.append(firstClassY,y)
    else:
        secondClassX,secondClassY = np.append(secondClassX,x),np.append(secondClassY,y)

fig = plt.figure()

fig = plt.plot(firstClassX,firstClassY,"r.",label="Κλάση 1")
fig = plt.plot(secondClassX,secondClassY,"bo",label="Κλάση 2")

fig = plt.legend()


##Δεύτερο γράφημα // διαδικασία εκπαίδευσης

a = Adaline(lr=0.02,epochs=Epochs)
a.fit(x_train,y_train,2)



##Τρίτο γράφημα

for index in range(Epochs):
    ##Εκπαίδευση Adaline
    a = Adaline(lr=0.005,epochs=index+1)
    a.fit(x_train,y_train)
    predictions = a.predict(x_train)
    
    ##Εκκίνηση Αποθήκευσης
    
    showedARanger,showedBRanger,showedA,showedB = np.array([]),np.array([]),np.array([]),np.array([])
    isARanger,isBRanger,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    
    
    
    for i in range(len(x_train)):
        temp = predictions[i]
        tempReal = y_train[i]
        
        
        if (tempReal==0):
            isARanger,isA = np.append(isARanger,i),np.append(isA,tempReal)
            showedARanger,showedA = np.append(showedARanger,i),np.append(showedA,temp)            
        else:
            isBRanger,isB = np.append(isBRanger,i),np.append(isB,tempReal)
            showedBRanger,showedB = np.append(showedBRanger,i),np.append(showedB,temp)

       
    #Φτιάχνουμε τα γραφήματα
    fig3 = plt.figure()
    fig3 = plt.plot(showedARanger,showedA,"mx",label="Πρόβλεψε 0")
    fig3 = plt.plot(showedBRanger,showedB,"gx",label="Πρόβλεψε 1")
    fig3 = plt.plot(isARanger,isA,"mo",label="Είναι 0",MarkerFaceColor='none')
    fig3 = plt.plot(isBRanger,isB,"go",label="Είναι 1",MarkerFaceColor='none')
    fig3 = plt.title("Εποχή: "+str(index+1))
    
    plt.legend()
    
##Τέταρτο γράφημα

for index in range(Epochs):
    ##Εκπαίδευση Perceptron
    a2 = Adaline(lr=0.005,epochs=index+1)
    a2.fit(x_train,y_train)
    predictions = a.predict(x_train)
    
    ##Εκκίνηση Αποθήκευσης
    
    showedARanger,showedBRanger,showedA,showedB = np.array([]),np.array([]),np.array([]),np.array([])
    isARanger,isBRanger,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    
    
    
    for i in range(len(x_test)):
        temp = predictions[i]
        tempReal = y_test[i]
        
        
        if (tempReal==0):
            isARanger,isA = np.append(isARanger,i),np.append(isA,tempReal)
            showedARanger,showedA = np.append(showedARanger,i),np.append(showedA,temp)            
        else:
            isBRanger,isB = np.append(isBRanger,i),np.append(isB,tempReal)
            showedBRanger,showedB = np.append(showedBRanger,i),np.append(showedB,temp)

       
    #Φτιάχνουμε τα γραφήματα
    fig4 = plt.figure()
    fig4 = plt.plot(showedARanger,showedA,"mx",label="Πρόβλεψε 0")
    fig4 = plt.plot(showedBRanger,showedB,"gx",label="Πρόβλεψε 1")
    fig4 = plt.plot(isARanger,isA,"mo",label="Είναι 0",MarkerFaceColor='none')
    fig4 = plt.plot(isBRanger,isB,"go",label="Είναι 1",MarkerFaceColor='none')
    fig4 = plt.title("ΤΕΣΤ, Εποχή: "+str(i+1))
    
    plt.legend()
    
##Πέμπτο γράφημα
cost,costIndex = a2.cost_,np.arange(len(a2.cost_))


fig5 = plt.figure()
fig5 = plt.plot(costIndex,cost,"b-")
fig5 = plt.title("Τετραγωνικό σφάλμα goes brrrr")
plt.xlim([0,10])


