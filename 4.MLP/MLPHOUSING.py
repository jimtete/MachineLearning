import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

file_path = "./housing.data"
housing_data = pd.read_csv(file_path)
housing_data_arr = housing_data.to_numpy()

##Τριμάρουμε τα άρχρηστα spaces και τα κάνουμε :
for i in range(len(housing_data_arr)):
    housing_data_arr[i,0]=(housing_data_arr[i,0]).replace("  "," ")
    housing_data_arr[i,0]=(housing_data_arr[i,0]).replace("  "," ")
    housing_data_arr[i,0]=((housing_data_arr[i,0]).replace(" ",":"))[1:]
    

##Μετατροπή των String σε float values
housing_data_arr_of_arr = np.zeros((len(housing_data_arr),14))
for i in range(len(housing_data_arr)):
    housing_data_arr_of_arr[i,:] = (housing_data_arr[i,0]).split(":")

data,dataValues=housing_data_arr_of_arr[:,0:13],housing_data_arr_of_arr[:,13]


x_train, x_test, y_train, y_test = train_test_split(data,dataValues, test_size=0.2)

Epochs = 69

for index in range(Epochs):
    mlp = MLPRegressor(random_state=1, max_iter=index+1,hidden_layer_sizes=(100,100))
    mlp = mlp.fit(x_train,y_train)
    predicted_y = mlp.predict(x_train)
    loss = mlp.loss_curve_
    beta = 1 
    ##Γράφημα 1
    
    fig = plt.figure()
    fig = plt.plot(np.arange(1,len(x_train)+1-beta),predicted_y[:-beta],"ro",label="Προβλεπούμενη τιμή")
    fig = plt.plot(np.arange(1,len(x_train)+1-beta),y_train[:-beta],"b.",label="Πραγματική τιμή"
                   ,MarkerFaceColor='none')
    
    fig = plt.title("Γράφημα 1: Στόχοι και προβλέψεις τιμών ακινήτων εποχή: "+str(index))
    fig = plt.xlabel("n:σπίτι")
    fig = plt.ylabel("τιμή σε *1000$")
    plt.legend()
    plt.show()
    
    
for index in range(Epochs):
    mlp = MLPRegressor(random_state=1, max_iter=index+1)
    mlp = mlp.fit(x_train,y_train)
    predicted_y = mlp.predict(x_train)
    loss = mlp.loss_curve_
    beta = 1 
    
    
    ##Γράφημα 2
    fig2 = plt.figure()
    fig2 = plt.plot(np.arange(1,len(loss)+1),loss,"b-",label="Μέσο τετραγωνικό σφάλμα")
    fig2 = plt.title("Γράφημα 2: Μέσο τετραγωνικό σφάλμα ανά εποχή")
    fig2 = plt.xlabel("Αιρθμός εποχών")
    fig2 = plt.ylabel("Ποσοστό σφάλματος")
    
    plt.legend()
    plt.show()
    
predicted_y_test = mlp.predict(x_test)
##Γράφημα 3
   
fig3 = plt.figure()
fig3 = plt.plot(np.arange(1,len(x_test)+1),predicted_y_test,"ro",label="Προβλεπούμενη τιμή")
fig3 = plt.plot(np.arange(1,len(x_test)+1),y_test,"b.",label="Πραγματική τιμή"
               ,MarkerFaceColor='none')

fig3 = plt.title("Γράφημα 3: Στόχοι και προβλέψεις τιμών ακινήτων - testset")
fig3 = plt.xlabel("n:σπίτι")
fig3 = plt.ylabel("τιμή σε *1000$")
plt.legend()
plt.show()



