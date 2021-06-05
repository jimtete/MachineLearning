import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn import datasets

##Εισαγωγή δεδομένων

iris = datasets.load_iris()
iris_data = iris.data[:,[0,2]]
iris_label = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris_data,iris_label, test_size=0.2)

##Πρώτο γράφημα

firstClassX,firstClassY = np.array([]),np.array([])
secondClassX,secondClassY = np.array([]),np.array([])
thirdClassX,thirdClassY = np.array([]),np.array([])

for i in range(len(x_train)):
    coords = x_train[i]
    x,y = coords
    if (y_train[i]==0):
        firstClassX,firstClassY=np.append(firstClassX,x),np.append(firstClassY,y)
    elif (y_train[i]==2):
        thirdClassX,thirdClassY=np.append(thirdClassX,x),np.append(thirdClassY,y)
    else:
        secondClassX,secondClassY=np.append(secondClassX,x),np.append(secondClassY,y)

fig1 = plt.figure()

fig1 = plt.title("Γράφημα 1 : τα αποτελέσματα στον χώρο των 2 διαστάσεων")
fig1 = plt.plot(firstClassX,firstClassY,"r.",label="Iris-setosa")
fig1 = plt.plot(secondClassX,secondClassY,"go",label="Iris-versicolor")
fig1 = plt.plot(thirdClassX,thirdClassY,"bx",label="Iris-virginica")

plt.legend()
plt.show()

Epochs=25

predictionClasses=np.zeros((Epochs,120))
predictionRealClasses = np.zeros((Epochs,120))

predictionClassesTest = np.zeros((Epochs,30))
predictionClassesTestReal = np.zeros((Epochs,30))
for index in range(Epochs):
    svm = SVC(kernel='rbf',max_iter=Epochs)
    svm = svm.fit(x_train,y_train)
    
    predictions_train = svm.predict(x_train)
    predictions_test = svm.predict(x_test)
    
    predictionClasses[index,:]=predictions_train
    predictionRealClasses[index,:]=y_train
    
    predictionClassesTest[index,:]=predictions_test
    predictionClassesTestReal[index,:]=y_test
    
    x0,y0 = np.array([]),np.array([])
    x1,y1 = np.array([]),np.array([])
    x2,y2 = np.array([]),np.array([])
    
    for i in range(len(predictions_train)):
        data_ = predictions_train[i]
        x = x_train[i,0]
        y = x_train[i,1]
        
        if (data_<0.5):
            x0 = np.append(x0,x)
            y0 = np.append(y0,y)
        elif(data_<1.5):
            x1 = np.append(x1,x)
            y1 = np.append(y1,y)
        else:
            x2 = np.append(x2,x)
            y2 = np.append(y2,y)
    
    fig2 = plt.figure()
    
    fig2 = plt.title("Γράφημα 2 : Οι προβλέψεις στον χώρο των 2 διαστάσεων, εποχή: "+str(index))
    fig2 = plt.plot(x0,y0,"r.",label="Iris-setosa")
    
    fig2 = plt.plot(x1,y1,"go",label="Iris-versicolor")
    fig2 = plt.plot(x2,y2,"bx",label="Iris-virginica")
    
    plt.legend()
    plt.show()
    
##Γράφημα 3: Σύγκριση μεταξύ Iris-setosa και Iris-versicolor
for i in range(Epochs):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    index=0
    for j in range(120):
        tempReal = predictionRealClasses[i,j]
        temp = predictionClasses[i,j]
        if (tempReal==0):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,index),np.append(SRA,index)
            index+=1
        elif(tempReal==1):
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,index),np.append(SRB,index)
            index+=1
    
    fig3 = plt.figure()
    fig3 = plt.plot(SRA,showedA,"mx",label="Πρόβλεψε Setosa")
    fig3 = plt.plot(SRB,showedB,"gx",label="Πρόβλεψε Versicolor")
    fig3 = plt.plot(IRA,isA,"mo",label="Είναι Setosa",MarkerFaceColor='none')
    fig3 = plt.plot(IRB,isB,"go",label="Είναι Versicolor",MarkerFaceColor='none')
    
    fig3 = plt.title("Γράφημα 3: Στόχοι και προβλέψεις ανά εποχή για setosa//versicolor: "+str(i))
    
    plt.legend()
    
##Γράφημα 4: Σύγκριση μεταξύ Iris-setosa και Iris-virginica
for i in range(Epochs):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    index=0
    for j in range(120):
        tempReal = predictionRealClasses[i,j]
        temp = predictionClasses[i,j]
        if (tempReal==0):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,index),np.append(SRA,index)
            index+=1
        elif(tempReal==2):
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,index),np.append(SRB,index)
            index+=1
    
    fig4 = plt.figure()
    fig4 = plt.plot(SRA,showedA,"mx",label="Πρόβλεψε Setosa")
    fig4 = plt.plot(SRB,showedB,"cx",label="Πρόβλεψε virginica")
    fig4 = plt.plot(IRA,isA,"mo",label="Είναι Setosa",MarkerFaceColor='none')
    fig4 = plt.plot(IRB,isB,"co",label="Είναι virginica",MarkerFaceColor='none')
    
    fig4 = plt.title("Γράφημα 4: Στόχοι και προβλέψεις ανά εποχή για setosa//virginica: "+str(i))
    
    plt.legend()

##Γράφημα 5: Σύγκριση μεταξύ Iris-versicolor και Iris-virginica
for i in range(Epochs):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    index=0
    for j in range(120):
        tempReal = predictionRealClasses[i,j]
        temp = predictionClasses[i,j]
        if (tempReal==1):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,index),np.append(SRA,index)
            index+=1
        elif(tempReal==2):
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,index),np.append(SRB,index)
            index+=1
    
    fig5 = plt.figure()
    fig5 = plt.plot(SRA,showedA,"gx",label="Πρόβλεψε versicolor")
    fig5 = plt.plot(SRB,showedB,"cx",label="Πρόβλεψε virginica")
    fig5 = plt.plot(IRA,isA,"go",label="Είναι versicolor",MarkerFaceColor='none')
    fig5 = plt.plot(IRB,isB,"co",label="Είναι virginica",MarkerFaceColor='none')
    
    fig5 = plt.title("Γράφημα 5: Στόχοι και προβλέψεις ανά εποχή για versicolor//virginica: "+str(i))
    
    plt.legend()
    
##Γράφημα 6: Σύγκριση μεταξύ Iris-setosa και Iris-versicolor
for i in range(Epochs):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    index=0
    for j in range(30):
        tempReal = predictionClassesTest[i,j]
        temp = predictionClassesTestReal[i,j]
        if (tempReal==0):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,index),np.append(SRA,index)
            index+=1
        elif(tempReal==1):
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,index),np.append(SRB,index)
            index+=1
    
    fig6 = plt.figure()
    fig6 = plt.plot(SRA,showedA,"mx",label="Πρόβλεψε Setosa")
    fig6 = plt.plot(SRB,showedB,"gx",label="Πρόβλεψε Versicolor")
    fig6 = plt.plot(IRA,isA,"mo",label="Είναι Setosa",MarkerFaceColor='none')
    fig6 = plt.plot(IRB,isB,"go",label="Είναι Versicolor",MarkerFaceColor='none')
    
    fig6 = plt.title("Γράφημα 6: Στόχοι και προβλέψεις ανά εποχή για setosa//versicolor (test): "+str(i))
    
    plt.legend()
    
##Γράφημα 7: Σύγκριση μεταξύ Iris-setosa και Iris-virginica
for i in range(Epochs):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    index=0
    for j in range(30):
        tempReal = predictionClassesTest[i,j]
        temp = predictionClassesTestReal[i,j]
        if (tempReal==0):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,index),np.append(SRA,index)
            index+=1
        elif(tempReal==2):
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,index),np.append(SRB,index)
            index+=1
    
    fig7 = plt.figure()
    fig7 = plt.plot(SRA,showedA,"mx",label="Πρόβλεψε Setosa")
    fig7 = plt.plot(SRB,showedB,"cx",label="Πρόβλεψε virginica")
    fig7 = plt.plot(IRA,isA,"mo",label="Είναι Setosa",MarkerFaceColor='none')
    fig7 = plt.plot(IRB,isB,"co",label="Είναι virginica",MarkerFaceColor='none')
    
    fig7 = plt.title("Γράφημα 7: Στόχοι και προβλέψεις ανά εποχή για setosa//virginica (test): "+str(i))
    
    plt.legend()
    
##Γράφημα 8: Σύγκριση μεταξύ Iris-versicolor και Iris-virginica
for i in range(Epochs):
    showedA,showedB,isA,isB = np.array([]),np.array([]),np.array([]),np.array([])
    IRA,IRB,SRA,SRB = np.array([]),np.array([]),np.array([]),np.array([])
    index=0
    for j in range(30):
        tempReal = predictionClassesTest[i,j]
        temp = predictionClassesTestReal[i,j]
        if (tempReal==1):
            isA,showedA = np.append(isA,tempReal),np.append(showedA,temp)
            IRA,SRA = np.append(IRA,index),np.append(SRA,index)
            index+=1
        elif(tempReal==2):
            isB,showedB = np.append(isB,tempReal),np.append(showedB,temp)
            IRB,SRB = np.append(IRB,index),np.append(SRB,index)
            index+=1
    
    fig8 = plt.figure()
    fig8 = plt.plot(SRA,showedA,"gx",label="Πρόβλεψε versicolor")
    fig8 = plt.plot(SRB,showedB,"cx",label="Πρόβλεψε virginica")
    fig8 = plt.plot(IRA,isA,"go",label="Είναι versicolor",MarkerFaceColor='none')
    fig8 = plt.plot(IRB,isB,"co",label="Είναι virginica",MarkerFaceColor='none')
    
    fig8 = plt.title("Γράφημα 8: Στόχοι και προβλέψεις ανά εποχή για versicolor//virginica (test): "+str(i))
    
    plt.legend()
    
