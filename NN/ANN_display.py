# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:16:08 2021

@author: dominique
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
plt.close('all')

#load ANN model
fname = 'ANN_model/20210629'
model = keras.models.load_model(fname)
testlist = np.load(fname+'/testlist.npy')

#load dataframe
df = pd.read_pickle('dataframe')

#seperation between features and labels
features = df
features = features.drop(columns=['PiF'])
features = features.drop(columns=['name'])
features = features.drop(columns=['weight'])
labels = features.pop('PiF_log')

#prediction of the ANN model
predictions = model.predict(features).flatten()

#get index of the test boundarylayer within the dataframe
def testIdx(df,trainlist):
    #function that extract idx of testing boundary layers
    lenfile = len(np.unique(df['name']))
    lenf = len(df)/lenfile
    #list of all the names
    namelist = np.unique(df['name'])
    idx = []
    for name in namelist:
        #check if name is in the train list
        if name in trainlist:
            print(name)
            #if not find first idx of the name
            list_idx = np.where(df['name']==name)
            #add idx to list
            idx.append(int(list_idx[0][0]/lenf))
        
    return idx
    
test_idx = testIdx(df,testlist)

def plot_model(df,idx):
    lenfile = len(np.unique(df['name']))
    lenf = len(df)/lenfile
    lgc_start = int(idx*lenf)
    lgc_end = int((idx+1)*lenf)-1
    
    
    predictions = model.predict(features).flatten()
    
    
    plt.figure()
    plt.semilogx(df.iloc[lgc_start:lgc_end]['Pi1'],df.iloc[lgc_start:lgc_end]['PiF_log'],linestyle='',marker='.',color='b',label='data')
    plt.semilogx(df.iloc[lgc_start:lgc_end]['Pi1'],predictions[lgc_start:lgc_end],linestyle='-',color='r',label='NN')
    
    plt.grid()
    plt.xlabel(r"$ \omega \delta^* /U_e $ ")
    plt.ylabel(r"$ \Phi_{pp}  U_e / \tau_w^2 \delta^*$ ")
    plt.legend()
    plt.axis([0.01, 100, -70, 30])
    plt.show()
    

#plot test case and compare to existing models
for i in test_idx:
    plot_model(df,i)
