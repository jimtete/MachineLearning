# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:14:47 2021

@author: Dimitris
"""
import pandas as pd
import numpy as np
import random
"""Creating the data 0.0:0.3 // 0.7:0.9"""


n=200
numbers = np.zeros(shape=(n,2))
classes = np.zeros(shape=(n))


for i in range(n):
    key =  (random.randint(0, 1))
    if (key==1):
        numbers[i] = [(random.uniform(0.7,0.9)),(random.uniform(0.7,0.9))]
    else:
        numbers[i] = [(random.uniform(0.0,0.3)),(random.uniform(0.0,0.3))]
    classes[i] = key
    
    
print(numbers[:,0],numbers[:,1])
raw_data = {'X': numbers[:,0],
            'Y': numbers[:,1]}

raw_data_values = {'values':classes}

df = pd.DataFrame(raw_data, columns = ['X','Y'])
df.to_csv('./exported_data_pack_1.csv')

df2 = pd.DataFrame(raw_data_values, columns = ['values'] )
df2.to_csv('./exported_data_pack_values_1.csv')

    
"""End creation of data"""

"""Creating the data [2]"""


n=200
numbers = np.zeros(shape=(n,2))
classes = np.zeros(shape=(n))


for i in range(n):
    key =  (random.randint(0, 2))
    if (key==1):
        numbers[i] = [(random.uniform(0.4,0.9)),(random.uniform(0.0,0.9))]
        classes[i] = 1
    elif (key==2):
        numbers[i] = [(random.uniform(0.0,0.3)),(random.uniform(0.4,0.9))]
        classes[i] = 1
    else:
        numbers[i] = [(random.uniform(0.0,0.3)),(random.uniform(0.0,0.3))]
        classes[i] = -1
    
    

"""End creation of data"""

"""Creating the data [3]"""


n=200
numbers = np.zeros(shape=(n,2))
classes = np.zeros(shape=(n))


for i in range(n):
    key =  (random.randint(0,7))
    if (key==0):
        numbers[i] = [(random.uniform(0.0,0.9)),(random.uniform(0.0,0.3))]
        classes[i] = 1
    elif (key==1):
        numbers[i] = [(random.uniform(0.0,0.9)),(random.uniform(0.7,0.9))]
        classes[i] = 1
    elif (key==1):
        numbers[i] = [(random.uniform(0.0,0.3)),(random.uniform(0.0,0.9))]
        classes[i] = 1
    elif (key==1):
        numbers[i] = [(random.uniform(0.7,0.9)),(random.uniform(0.0,0.9))]
        classes[i] = 1
    else:
        numbers[i] = [(random.uniform(0.4,0.6)),(random.uniform(0.4,0.6))]
        classes[i] = -1
    
    

"""End creation of data"""