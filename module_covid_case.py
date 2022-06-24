# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:11:00 2022

@author: diviyah
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input


class EDA():
    def __init__(self): 
        pass
    
    def plot_graph(self,df):
        plt.figure()
        plt.plot(df['cases_new'])
        plt.plot(df['cases_import'])
        plt.plot(df['cases_recovered'])
        plt.plot(df['cases_cluster'])
        plt.legend(['cases_new','cases_import','cases_recovered','cases_cluster'])
        plt.title('Covid-19 cases in Malaysia')
        plt.show()
        
class ModelCreation():       
    def __init__(self):
        pass
    
    def simple_lstm_model(self,X_train,num_node=16,drop_rate=0.05,activation='relu',
                          output_node=1):
        
        model = Sequential()
        model.add(Input(shape=(np.shape(X_train)[1],1)))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation=activation))
        
        model.summary()
        
        return model
    
class ModelEvaluation():
    def __init__(self): 
        pass
    
    
    def plot_hist(self,hist):
        
        hist.history.keys()

        plt.figure()
        plt.plot(hist.history['mape'])
        plt.legend('mape')
        plt.show()   #MAPE flunctuates like crazy at first

        plt.figure()
        plt.plot(hist.history['loss'])
        plt.legend('loss')
        plt.show()   # mse reduces further 
        
        
        
    def plot_predicted_graph(self,test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='Actual Number of Covid Cases')
        plt.plot(predicted,'r',label='Actual Number of Covid Cases')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='Actual Number of Covid Cases')
        plt.plot(mms.inverse_transform(predicted),'r',label='Actual Number of Covid Cases')
        plt.legend()
        plt.show()