# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:16:08 2021

@author: dominique
"""
import numpy as np
import pandas as pd
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

def plot_model(df,idx):
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
    
    #prediction of the ANN
    predictions = model.predict(features).flatten()
    
    
    textstr = '\n'.join((
    r'$Delta=%.2f$' % (blparam['delta']/blparam['delta_star'], ),
    r'$H=%.2f$' % (blparam['delta_star']/blparam['theta'], ),
    r'$Ma=%.2f$' % (blparam['Ue']/343, ),
    r'$Pi=%.2f$' % (blparam['Picoles'], ),
    r'$Cf=%.2f \times 10 ^{-3}$' % (blparam['Cf']*1000, ),
    r'$RT=%.2f$' % ((blparam['delta_star']/blparam['Ue'])/(1.562e-5/blparam['Utau']**2), ),
    r'$Beta=%.2f$' % (blparam['betaC'], ),))
    
    
    
    fig, ax = plt.subplots()
    
    props = dict(boxstyle='round', alpha=0.5)
    
    plt.semilogx(df.iloc[lgc_start:lgc_end]['Pi1'],df.iloc[lgc_start:lgc_end]['PiF_log'],linestyle='',marker='.',color='b',label='data')
    plt.semilogx(df.iloc[lgc_start:lgc_end]['Pi1'],predictions[lgc_start:lgc_end],linestyle='-',color='r',label='NN')
    plt.grid()
    plt.xlabel(r"$\omega \delta^* / U_e$  ")
    plt.ylabel(r"$ \Phi_{pp}  U_e / \tau_w^2 \delta^*$ ")
    #plt.title(df['name'].iloc[lgc_start])
    plt.legend()
    plt.axis([0.01, 100, -70, 30])
    ax.text(0.05, 0.50, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.show()
    

#plot test case and compare to existing models
for i in test_idx:
    plot_model(df,i)