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
data = pd.read_csv(Dedomena2)
dataValues = pd.read_csv(Dedomena2Values)

x_train, x_test, y_train, y_test = train_test_split(data, dataValues, test_size=0.2)
x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

##Πρώτο γράφημα

firstClassX,firstClassY,firstClassZ = np.array([]),np.array([]),np.array([])
secondClassX,secondClassY,secondClassZ = np.array([]),np.array([]),np.array([])

for i in range(len(x_train)):
    coords = x_train[i]
    x,y,z = coords
    if (y_train[i]==0):
        firstClassX,firstClassY,firstClassZ=np.append(firstClassX,x),np.append(firstClassY,y),np.append(firstClassZ,z)
    else:
        secondClassX,secondClassY,secondClassZ=np.append(secondClassX,x),np.append(secondClassY,y),np.append(secondClassZ,z)

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')

fig1 = ax.scatter(firstClassX,firstClassY,firstClassZ,label="Κλάση 0", c='red', marker='x')
fig1 = ax.scatter(secondClassX, secondClassY, secondClassZ,label="Κλάση 1", c='blue', marker='o')
fig1 = ax.set_xlabel('x')
fig1 = ax.set_ylabel('y')
fig1 = ax.set_zlabel('z')
fig1 = plt.title("Γράφημα 1: Δεδομένα στον χώρο των τριών διαστάσεων")



plt.legend()

plt.show()

##Οιρσμός εποχών
Epochs=100
amountOfData = 50
##Δημιουργία MLP
mlp = MLPClassifier(verbose=False,learning_rate_init=0.04,max_iter=Epochs,
                   activation='tanh', solver = 'sgd',hidden_layer_sizes=(6))

mlp.partial_fit(x_train,y_train,classes=np.unique(y_train))

predictionClasses = np.zeros((Epochs-1,50))
predictionRealClasses = np.zeros((Epochs-1,50))

for index in range(Epochs-1):
    mlp.partial_fit(x_train,y_train,classes=None)
    
    predictions_train = mlp.predict(x_train)
    x0,y0,z0 = np.array([]),np.array([]),np.array([])
    x1,y1,z1 = np.array([]),np.array([]),np.array([])
    
    
    for i in range(len(predictions_train)):
        data_ = predictions_train[i]
        
        if (i<amountOfData):
            predictionClasses[index,i]=data_
            predictionRealClasses[index,i]=y_train[i]
        
        x = x_train[i,0]
        y = x_train[i,1]
        z = x_train[i,2]
        
        if (data_<0.5):
            #print(x,y,z," is red")
            x0 = np.append(x0,x)
            y0 = np.append(y0,y)
            z0 = np.append(z0,z)
        elif (data_>0.5):
            #print(x,y,z," is blue")
            x1,y1,z1 = np.append(x1,x),np.append(y1,y),np.append(z1,z)
            
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    
    fig2 = ax.scatter(x0,y0,z0,label="Κλάση 0", c='red', marker='x')
    fig2 = ax.scatter(x1, y1, z1,label="Κλάση 1", c='blue', marker='o')
    fig2 = ax.set_xlabel('x')
    fig2 = ax.set_ylabel('y')
    fig2 = ax.set_zlabel('z')
    fig2 = plt.title("Γράφημα 2: Προβλέψεις δεδομένων εποχή: "+str(index))
        
    plt.legend()

    plt.show()  

###Τρίτο γράφημα ( για τα πρώτα 50 δεδομένα ) var amountOfData


for i in range(Epochs-1):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    for j in range(amountOfData):
        tempReal = predictionRealClasses[i,j]
        temp = predictionClasses[i,j]
        if (tempReal==0):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,j),np.append(SRA,j)
            
        else:
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,j),np.append(SRB,j)
           
    fig3 = plt.figure()
    fig3 = plt.plot(SRA,showedA,"mx",label="Πρόβλεψε 0")
    fig3 = plt.plot(SRB,showedB,"gx",label="Πρόβλεψε 1")
    fig3 = plt.plot(IRA,isA,"mo",label="Είναι 0",MarkerFaceColor='none')
    fig3 = plt.plot(IRB,isB,"go",label="Είναι 1",MarkerFaceColor='none')
    
    fig3 = plt.title("Γράφημα 3: Στόχοι και προβλέψεις ανά εποχή για τα πρώτα 50 δεδομένα: "+str(i))
    
    plt.legend()
    
    plt.show()

##Τέταρτο γράφημα
lossCurve = mlp.loss_curve_

fig4 = plt.figure()
fig4 = plt.plot(np.arange(len(lossCurve)),lossCurve,"b--.")
fig4 = plt.xlabel("Αιρθμός εποχών")
fig4 = plt.ylabel("Ποσοστό σφάλματος")
fig4 = plt.title("Γράφημα 4: Ποσοστό σφάλματος / εποχή")
plt.legend()

plt.show()


##Πέμπτο γράφημα

a=y_train


predTest = mlp.predict(x_test)

b=y_test



x0,y0,z0 = np.array([]),np.array([]),np.array([])
x1,y1,z1 = np.array([]),np.array([]),np.array([])

d1,d2 = x_test,y_test

for s in range(len(predTest)):
    
    x = x_test[s,0]
    y = x_test[s,1]
    z = x_test[s,2]
    
    
    
    
    
    
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



    