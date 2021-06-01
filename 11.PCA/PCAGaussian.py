import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

dataTemp = np.load('./mnist_49.npz', mmap_mode='r')

data = dataTemp['x']
dataValues = dataTemp['t']

num_components = np.array([1,2,5,10,20,30,40,50,60,70,80,100,200,500,784])
acc_train = np.array([])
acc_test = np.array([])

for iC, C in enumerate(num_components):
    pca = PCA(n_components=C)
    x_pca = pca.fit_transform(data)
    
    x_train, x_test, y_train, y_test = train_test_split(x_pca, dataValues, test_size=0.2)
    
    GNB = GaussianNB()
    GNB.fit(x_train,y_train)
    
    logTrain = GNB.score(x_train,y_train)
    logTest = GNB.score(x_test,y_test)
    
    acc_train = np.append(acc_train,logTrain)
    acc_test = np.append(acc_test,logTest)
    
##Πρώτο γράφημα

fig = plt.figure()
fig = plt.plot(num_components,acc_train,"c-",label="Train")
fig = plt.plot(num_components,acc_test,"g--",label="Test")
fig = plt.xlabel("Αριθμός βαρών, num_components")
fig = plt.ylabel("Ποσοστό Acurracy")
plt.xlim([1,784])
#plt.ylim([0.4,1])

plt.legend()
plt.show()