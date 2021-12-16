# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:53:44 2021

@author: dominique
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
plt.close('all')

#load dataframe of data
df = pd.read_pickle('dataframe')

#transform to decibel
def log_op(x):
    return 10*np.log10(x)

df['PiF_log'] = df['PiF'].apply(log_op)

# chose profile to be removed and keept for testing
namelist = ['2014_salze_case_zpg_U_45','2020_hao_case_CD_xc_0.02','2020_deuse_case_CD_xc_0.02',
            '2010_christophe_case_3_xc_0.05','2010_christophe_case_1_xc_0.20','2010_christophe_case_5_xc_0.10']

def data_partition_custum(df,namelist):
    # function that seperate testing form training/validation data
    sample = namelist
    
    test_dataset = pd.DataFrame()
    for ii in range(len(sample)):
        data = df.loc[df['name'] == sample[ii]]
        test_dataset = test_dataset.append(data)


    train_dataset = df.drop(test_dataset.index)

    #separate labels form features
    train_features = train_dataset.copy()
    train_features = train_features.drop(columns=['PiF'])
    train_features = train_features.drop(columns=['name'])
    weight_sample = train_features['weight']
    train_features = train_features.drop(columns=['weight'])
    
    test_features = test_dataset.copy()
    test_features = test_dataset.drop(columns=['PiF'])
    test_features = test_features.drop(columns=['name'])
    test_features = test_features.drop(columns=['weight'])
    
    train_labels = train_features.pop('PiF_log')
    test_labels = test_features.pop('PiF_log')
    
    return train_features, train_labels, test_features, test_labels, sample, weight_sample

#spit testing / training and validation data
features, labels, test_features, test_labels, trainlist, weight_sample = data_partition_custum(df,namelist)
#assign weight
features['weight']  = weight_sample

#split testing / training data
train_features, validate_features, train_labels, validate_labels = train_test_split(features,labels,test_size=0.2)

#assign weight and features to new array
train_weight= train_features['weight']
train_features = train_features.drop(columns=['weight'])
validation_weight = validate_features['weight']
validate_features = validate_features.drop(columns=['weight'])

# Normalize input 
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# Build Model
def build_and_compile_model(norm):
    model = keras.Sequential([norm,
                            layers.Dense(10, activation='selu',kernel_initializer='lecun_uniform'),
                            layers.Dense(10, activation='selu',kernel_initializer='lecun_uniform'),
                            layers.Dense(10, activation='selu',kernel_initializer='lecun_uniform'),
                            layers.Dense(1)
                            ])

    model.compile(loss="MeanSquaredError",
                  optimizer=keras.optimizers.Nadam(0.0001))
    return model

dnn_model = build_and_compile_model(normalizer)

#get number of trainable parameters
model_sum = dnn_model.summary()

#impose early stoping
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights =1)

#training model
history = dnn_model.fit(train_features, train_labels,
                        validation_data=(validate_features, validate_labels, validation_weight),
                        batch_size = 32, #round(len(train_dataset)/4),
                        sample_weight = train_weight,
                        callbacks=[callback],
                        verbose=True, epochs=10000)

#plot model training
def plot_loss(history):
    
    min_Idx = np.argmin(history.history['val_loss'])
    
    plt.figure()
    plt.plot(history.history['loss'],label='training loss', linewidth=4)
    plt.plot(history.history['val_loss'],linestyle = '--', label='validation loss')
    plt.plot(min_Idx,history.history['val_loss'][min_Idx], color ='r', marker ='d', label='best iteration')
    plt.xlabel('Iteration',fontsize=22)
    plt.ylabel('lMSE',fontsize=22)
    plt.legend()
    plt.grid(True)
    
plot_loss(history)

#show prediction on the training set
train_predictions = dnn_model.predict(train_features).flatten()
fit_train = np.mean((train_predictions-train_labels)**2)

plt.figure()
plt.semilogx(train_features["Pi1"],train_labels,linestyle='',marker='.',color='b')
plt.semilogx(train_features["Pi1"],train_predictions,linestyle='',marker='.',color='g')
plt.grid()
plt.title('training set - fit  =%.2f'%(fit_train))
plt.show()

#show prediction on the testing set
test_predictions = dnn_model.predict(test_features).flatten()
fit_test = np.mean((test_predictions-test_labels)**2)

plt.figure()
plt.semilogx(test_features["Pi1"],test_labels,linestyle='',marker='.',color='b')
plt.semilogx(test_features["Pi1"],test_predictions,linestyle='',marker='.',color='r')
plt.grid()
plt.title('testing set - fit  =%.2f'%(fit_test))
plt.show()


#save model (optional)
save_model = 0
model_name = 'whatever_you_like'
if save_model:
    fpath = 'wps_model/whatever_you_like'
    dnn_model.save(fpath)
    np.save(fpath + '/testlist.npy',trainlist)



