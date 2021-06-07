from matplotlib import image as img
from matplotlib import pyplot as plt
import numpy as np
#from hopfieldnetwork import HopfieldNetwork
from neurodynex3.hopfield_network import network, pattern_tools

##Εισαγωγή δεδομένων

##Δημιουργία πίνακα αριθμών
numbers = np.array((5,6,8,9))


##Δημιουργία Πινάκων δεδομένων
perfectData = np.zeros((len(numbers),11,7,4))
perfectDataFlattened = np.zeros((len(numbers),11*7))
perfectDataValues = numbers

##Διάβασμα των φωτογραφιών η αποθήκευσή τους.
index=0
for i in numbers:
    file_in = "./bitmaps/perfect/{}.png".format(i)
    tempimg = img.imread(file_in)
    perfectData[index] = tempimg
    index+=1
  
##temp δοκιμαστικό
arithmos="5" #5,6,8,9
paradeigma="5" #1,2,3,4 -- 5,6
temp = img.imread("./bitmaps/imperfect/{}-{}.png".format(arithmos,paradeigma))
tempData= temp[:,:,0]
tempData = np.where(tempData==1,-1,tempData)
tempData = np.where(tempData>0,1,tempData)
plt.imshow(tempData)
plt.show()


##Μετατροπή των δεδομένων
perfectDataTemp = perfectData[:,:,:,0]
perfectDataTemp = np.where(perfectDataTemp==1,-1,perfectDataTemp)
perfectDataTemp = np.where(perfectDataTemp>0,1,perfectDataTemp)
perfectData = perfectDataTemp

'''
for i in range(0,4):
    plt.imshow(perfectData[i])
    plt.show()
'''
    
##Επιπέδωση των δεδομένων
for i in range(0,4):
    perfectDataFlattened[i] = perfectData[i].flatten()

tempData = tempData.flatten()
##Εκπαίδευση Δικτύου


networkHopfield = network.HopfieldNetwork(nr_neurons=77)
networkHopfield.store_patterns(perfectDataFlattened)
networkHopfield.set_state_from_pattern(tempData)


states = networkHopfield.run_with_monitoring(4)
factory = pattern_tools.PatternFactory(11, 7)
states_as_patterns = factory.reshape_patterns(states)

dataValues = (np.asarray(states_as_patterns))

for i in range(4):
    plt.imshow(dataValues[i,:,:])
    plt.show()









#networkHopfield.update_neurons(iterations=5, mode=Mode)

#print(networkHopfield.compute_energy(tempData))

#networkHopfield.save_network("./network.npz")

''' for deletion
for i in range(0,4):
    networkHopfield.train_pattern(perfectDataFlattened[i])
    print(networkHopfield.compute_energy(perfectDataFlattened[i]))
'''