# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:38:50 2020

@author: dominique
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

def Goody_data(f, Rt, plt_fig=1, noise=1):
    #function that create a dataframe of WPS based on the goody model 
    
    #initialize dataframe
    Pi1 = [] #frequency
    Pi2 = [] #Rt
    PiF = [] #phipp

    #create boundary layer object
    bl = BLProf()
    
    plt.figure()
    for ii in range(len(Rt)):
        # Create a boundray layer object
        bl.blparam = {'rho_0':1.0,
                      'delta': 1.0,
                      'Ue': 1.0,
                      'tauw': 1.0,
                      'Utau': 1.0}
    
        # impose Rt through nu
        nu = 1/Rt[ii]
        
        # Compute Goody model
        Phipp = Phipp_Goody(bl,f,nu=nu)
        
        #append results
        
        idx = Phipp > 0.0005
        Pi1.append(f[idx])
        Pi2.append(Rt[ii]*np.ones(len(f[idx])))
        
      
        
        if noise ==1:   
            #noise
            Noise1 = np.random.normal(scale= 0.1, size=len(f[idx]))*Phipp[idx] 
            # background noise
            Noise2 = np.random.normal(scale= 0.001, size=len(f[idx])) 
        
            PiF.append(np.abs(Phipp[idx]  + Noise1 + Noise2))
        else:
            PiF.append(Phipp[idx])
            
        #test plot
        plt.semilogx(f[idx] ,10*np.log10(Phipp[idx] ))
        
    if plt_fig==1:
        plt.show()   
        
    # Construc dataframe
    PiF_final = np.concatenate(PiF, axis=0 )
    Pi1_final = np.concatenate(Pi1, axis=0 )
    Pi2_final = np.concatenate(Pi2, axis=0 )
    
    d = {'Pi1' : Pi1_final,
         'Pi2' : Pi2_final,
         'PiF': PiF_final}

    df = pd.DataFrame(data=d)
    
    #plot
    if plt_fig == 1:
        plt.figure()
        plt.semilogx(df["Pi1"],10*np.log10(df["PiF"]),marker='.',linestyle=' ')
        plt.show()
    return df 