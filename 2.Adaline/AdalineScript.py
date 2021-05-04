import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from Adaline import Adaline
import pdr


# Παίρνω τα δεδομένα
df = pd.read_csv('./iris.data',header=None)

# Πλοττάρω τα πρώτα 100 δεδομένα (setosa, versicolor)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()

# Κανονικοποιούμε τα δεδομένα
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

maxEpochs=150

for index in range(maxEpochs+1):
    # Δημιουργούμε ένα δίκτυο Adaline
    model1 = Adaline(lr=0.001,epochs=index)
    
    # Εκπαιδεύουμε το μοντέλο
    model1.fit(X_std, y)
       
    # Υπολογίζουμε την γραμμή
    pdr.plot_decision_regions(X_std, y, classifier = model1)
    plt.title('Adaline - Gradient Descent | epoch: '+str(index))
    plt.xlabel('sepal length [standardized')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()

# Υπολογίζουμε τα Errors
plt.plot(range(1, len(model1.cost_) + 1), model1.cost_, marker = 'o', color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()