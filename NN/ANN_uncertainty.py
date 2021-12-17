# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:16:08 2021

@author: dominique
"""
import numpy as np
import pandas as pd
from tensorflow import keras
import os

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
            #if not find first idx of the name
            list_idx = np.where(df['name']==name)
            #add idx to list
            idx.append(int(list_idx[0][0]/lenf))
        
    return idx
    
test_idx = testIdx(df,testlist)

#get boundray layer parameters from dataframe
def getBl(Pi2,Pi3,Pi4,Pi5,Pi6,Pi7,Pi8):
    
    rho = 1.1839
    nu = 1.562e-5
    
    Ue = Pi4*343 #equilibirum velocity
    Cf = Pi6 #friction coefficient
    tauw = 0.5*rho*Ue**2*Cf #wall shear stress
    utau = np.sqrt(tauw/rho) #friction velocity
    deltas = Pi7*(nu/utau**2)*Ue #displacement thickness
    delta = deltas/Pi2 #boundray layer thickness
    theta = Pi3*delta #momentum thickness
    dpds =  (Pi8 - 1)*tauw/theta #static pressure gardient
    PiCole = Pi5 #wake parameter
    
    blparam = {
        'Ue' : Ue, #[m/s]
        'Cf' : Cf, #[m/s]
        'tauw' : tauw, #[m/s]
        'delta' : delta, #[m]
        'delta_star' : deltas,
        'theta' : theta,
        'Utau' : utau, #[m/s]
        'betaC' : Pi8 - 1,
        'Picoles' : PiCole #[-]
        }

    return blparam

def point_uncertainty(df):
    
    #evaluate each model at the test grid
    path = "Ensemble/"
    dnnlist = os.listdir(path)
    
    #initialize output file
    output =np.zeros((len(dnnlist),len(df)))
    
    #loop on all models
    for i, fname in enumerate(dnnlist):
        print('\r %s - %i/%i'%(fname,i+1,len(dnnlist)), sep=' ', end='', flush=True)
        #print(' %s - %i/%i'%(fname,i+1,len(dnnlist)))
        #load model
        model = keras.models.load_model(path+'/'+fname)
        #predcit values
        predictions = model.predict(df).flatten()
        #save output
        output[i,:] = predictions
        
    
    return output

def plot_model(df,idx,error_in):
    #create idx vector of the relevant boundray layer
    lenfile = len(np.unique(df['name']))
    lenf = len(df)/lenfile
    lgc_start = int(idx*lenf)
    lgc_end = int((idx+1)*lenf)-1
    
    #get boundray layer parameters
    blparam = getBl(df['Pi2'].iloc[lgc_start],
                    df['Pi3'].iloc[lgc_start],
                    df['Pi4'].iloc[lgc_start],
                    df['Pi5'].iloc[lgc_start],
                    df['Pi6'].iloc[lgc_start],
                    df['Pi7'].iloc[lgc_start],
                    df['Pi8'].iloc[lgc_start])
    
    #create new frequency vector
    f = np.logspace(1, 7, num=1000)
    
    # compute one neural network prediction
    dtemp = {'Pi1' : 2*np.pi*blparam['delta_star']/blparam['Ue']*f,
         'Pi2' : df.iloc[lgc_start]['Pi2']*np.ones(len(f)),
         'Pi3' : df.iloc[lgc_start]['Pi3']*np.ones(len(f)),
         'Pi4' : df.iloc[lgc_start]['Pi4']*np.ones(len(f)),
         'Pi5' : df.iloc[lgc_start]['Pi5']*np.ones(len(f)),
         'Pi6' : df.iloc[lgc_start]['Pi6']*np.ones(len(f)),
         'Pi7' : df.iloc[lgc_start]['Pi7']*np.ones(len(f)),
         'Pi8' : df.iloc[lgc_start]['Pi8']*np.ones(len(f))}
    dNN = pd.DataFrame(data=dtemp)
    
    #compute ensemble average and standart deviation
    #epistemic uncertainty
    NN_uncertainty = point_uncertainty(dNN)
    #aleatory uncertainty
    sigma_in = error_in*np.ones(len(f))
    NN_uncertainty_std = np.sqrt(np.std(NN_uncertainty,0)**2 + sigma_in**2)
    #mean predictions
    NN_uncertainty_mean = np.mean(NN_uncertainty,0)
    

    plt.figure()
    plt.semilogx(df.iloc[lgc_start:lgc_end]['Pi1'],df.iloc[lgc_start:lgc_end]['PiF_log'],linestyle='',marker='.',color='b',label='data')
    plt.semilogx(dNN['Pi1'],NN_uncertainty_mean,linestyle='-',color='r',label='ANN',linewidth=3)
    plt.fill_between(dNN['Pi1'], NN_uncertainty_mean - 1.96*NN_uncertainty_std, NN_uncertainty_mean + 1.96*NN_uncertainty_std, color='salmon', alpha=0.5, label='ANN uncertainty')
    plt.grid()
    plt.xlabel(r"$ \omega \delta^* /U_e $ ")
    plt.ylabel(r"$ \Phi_{pp}  U_e / \tau_w^2 \delta^*$ ")
    #plt.title(df['name'].iloc[lgc_start])
    plt.legend()
    plt.axis([0.01, 100, -70, 30])

    plt.show()
    
#compute aleatory uncertainty
NN_in_prediction = point_uncertainty(features)

error = np.zeros(len(NN_in_prediction))
for i in range(len(NN_in_prediction)):
    error[i] = np.mean((NN_in_prediction[i,:] - labels)**2)

error_in = np.mean(error)  

#plot test case with uncertainties
for i in test_idx:
    plot_model(df,i,error_in)