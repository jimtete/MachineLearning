import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation and plotting
import matplotlib.pyplot as plt # data plotting
import warnings

# Seaborn default configuration
sns.set_style("darkgrid")

# set the custom size for my graphs
sns.set(rc={'figure.figsize':(8.7,6.27)})


warnings.filterwarnings('ignore') 


pd.options.display.max_columns=999 


import os
print(os.listdir("./"))

data = pd.read_csv("./iris.csv")

rows,col = data.shape
print("Rows : %s, column : %s" % (rows, col))


##Γενικά γραφήματα
snsdata = data.drop(['Id'], axis=1)
g = sns.pairplot(snsdata, hue='Species', markers='x')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)


##Εμφάνιση μεταβλητών
sns.violinplot(x='SepalLengthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='SepalWidthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='PetalLengthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='PetalWidthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()


##Θέτουμε τις μεταβλητές των δεδομένων

mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

X = data.drop(['Id', 'Species'], axis=1).values # Input Feature Values
y = data.Species.replace(mapping).values.reshape(rows,1) # Output values

X = np.hstack(((np.ones((rows,1))), X))# Adding one more column for bias

##Θέτουμε το b bias

np.random.seed(0) # Let's set the zero for time being
theta = np.random.randn(1,5) # Setting values of theta randomly

print("Theta : %s" % (theta))

##Μεταβλητές για την εκπαίδευση
epochs = 100
learning_rate = 0.003 # If you are going by formula, this is actually alpha.
J = np.zeros(epochs) # 1 x 10000 maxtix

# Εκπαιδεύουμε το μοντέλο
for i in range(epochs):
    J[i] = (1/(2 * rows) * np.sum((np.dot(X, theta.T) - y) ** 2 ))
    theta -= ((learning_rate/rows) * np.dot((np.dot(X, theta.T) - y).reshape(1,rows), X))
    prediction = np.round(np.dot(X, theta.T))
    ax = plt.subplot(111)

    ax.plot(np.arange(1, 151, 1), y, label='Πραγματική τιμή', color='red')
    ax.scatter(np.arange(1, 151, 1), prediction, label='Εκτίμηση δικτύου')
    
    ax.set_ylim([0,4])
    plt.xlabel("Dataset size", color="Green")
    plt.ylabel("Iris Flower (1-3)", color="Green")
    plt.title("Epoch: "+str(i)+" Iris Flower (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")
    
    ax.legend()
    plt.show() 



ax = plt.subplot(111)
ax.plot(np.arange(epochs), J)
ax.set_ylim([0,0.55])
plt.ylabel("Cost Values", color="Green")
plt.xlabel("No. of Iterations", color="Green")
plt.title("Mean Squared Error vs Iterations")
plt.show()

ax = sns.lineplot(x=np.arange(epochs), y=J)
plt.show()



accuracy = (sum(prediction == y)/float(len(y)) * 100)[0]
print("The model predicted values of Iris dataset with an overall accuracy of %s" % (accuracy))