# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:25:35 2022

@author: diviyah


"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
from module_covid_case import EDA,ModelCreation,ModelEvaluation


#%% STATIC

CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
MMS_PATH = os.path.join(os.getcwd(),'model','mms_covid_cases.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','saved_model.h5')
CSV_TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
log_dir=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH=os.path.join(os.getcwd(),'logs',log_dir)

#%% Data Loading

df = pd.read_csv(CSV_PATH)

#%% Data Inspection

df.info()  # The date and cases_new column is in object format
           # The cases_new has '?' in the data
           
temp=df.describe().T # mean of cases_new is very far off from the median. 
                     # The distribution of cases_new is not normally distributed

df.isna().sum() 

df.head()
   
EDA=EDA()
EDA.plot_graph(df)


#%% Data Cleaning

# TO DO - data needs to be converted into numeric for ['cases_new']
#       - Need to impute NaN by data interpolation

df['cases_new']=pd.to_numeric(df['cases_new'],errors='coerce')
df.info()  # cases_new has now become a float

df.isna().sum() # 12 missing values from cases_new

# Interpolation using polynomial with degree of 2 since the graph has 

df['cases_new'].interpolate(method='polynomial', order=2,inplace=True)

df.isna().sum()

#%% Feature Selection

# We are only selecting['cases_new']

# X=df['cases_new']

# Instead of selecting X variable first, we will be doing MinMax Scaling 
# or else will be very tedious
# So MinMaxScaling --> then extract the features out.

# Why MinMax not Standard?

# StandardScaler is useful for the features that follow a Normal distribution.
#..Therefore, it makes mean = 0 and scales the data to unit variance.
# MinMaxScaler may be used when the upper and lower boundaries are well known /
#..from domain knowledge.MinMaxScaler scales all the data features in the range
#.. [0, 1] or else in the range [-1, 1] if there are negative values in the 
#..dataset.This scaling compresses all the inliers in the narrow range.


#%% Data Preprocessing

mms=MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

# Saving the MinMaxScaled variable
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)
    
# this is to initialize the empty space- just to fill up.
# create something just like neural network. Need to declare first

X_train = []
y_train = []

win_size=30

len(df)
np.shape(df)[0] # the same as len(df)

# i=30
for i in range(win_size,np.shape(df)[0]): #Can write this way as well
# for i in range(win_size,X_train.shape(df)[0]): # BUT cant do this.. we need df
    X_train.append(df[i-win_size:i,0]) # WHAT IS THIS?
                                       # i-30 = 30-30 = 0 
                                       # so from [0:30] will give us?
                                       # it will give us first 30 elements
                                       # comma 0 means slicing, We are only taking the 1st col
                                       # and then we are appending it one by one
                                       # first day, we predict yn
                                       # second day we predict from i(n+1) to predict y(n+1)
    y_train.append(df[i,0]) #why y_train not deducting by anything, straight i je?
                            # Because we wanna predict the price the very next day
                            
X_train = np.array(X_train) #Why do we need to put np.array ya?
y_train = np.array(y_train) #Because all of x_train and y_train is np.list
                            # so we can add np.array to solve the problem
                            #No more list already
                            
                    
MC=ModelCreation()

model=MC.simple_lstm_model(X_train,num_node=16,drop_rate=0.05,activation='relu',
                      output_node=1)

model.compile(optimizer='adam',loss='mse',metrics='mape')


X_train = np.expand_dims(X_train,axis=-1)

# callbacks
tensorboard_callback=TensorBoard(log_dir=LOG_PATH)

hist = model.fit(X_train,y_train,
                  batch_size=32,
                  epochs=500,
                  callbacks=[tensorboard_callback])


#%% Model Evaluation 


ME=ModelEvaluation()
ME.plot_hist(hist)

#%% Model Architecture

plot_model(model,show_shapes=True, show_layer_names=(True))     

#%% Model Saving

model.save(MODEL_SAVE_PATH)

#%% Model Deployment and Analysis using Test data

test_df = pd.read_csv(CSV_TEST_PATH)

test_df.info() #cases_new is in object dtype

test_df['cases_new']=pd.to_numeric(test_df['cases_new'],errors='coerce')
test_df.info()  # cases_new has now become a float

test_df.isna().sum() #has one NaN

test_df['cases_new'].interpolate(method='polynomial', order=2,inplace=True)


test_df.isna().sum() #NONE!

test_df = mms.transform(np.expand_dims(test_df.iloc[:,1],axis=-1))

plt.figure()
plt.plot(test_df)
plt.show()

con_test = np.concatenate((df,test_df),axis=0)

con_test = con_test[-(win_size+len(test_df)):] #dynamic

X_test = []
for i in range (win_size, len(con_test)):
    X_test.append(con_test[i-win_size:i,0]) 

X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))   

#%% Plotting of Graphs 


# all of the data we have is scaled data
# Hence, need to inverse the data to see the real value

mms.inverse_transform(test_df)

ME.plot_predicted_graph(test_df, predicted, mms)



#%% MSE MAE MAPE

print((mean_absolute_error(test_df, predicted)/sum(abs(test_df)))*100)

mean_squared_error(test_df,predicted)
mean_absolute_error(test_df,predicted)

test_df_inversed = mms.inverse_transform(test_df)
predicted_inversed = mms.inverse_transform(predicted)

mean_squared_error(test_df_inversed,predicted_inversed)  #80.35710042920878
mean_absolute_error(test_df_inversed,predicted_inversed) #6.437275878906251
mean_absolute_percentage_error(test_df_inversed,predicted_inversed)


#%% DISCUSSION

# =============================================================================

# PROBLEM STATEMENT
# To create a deep learning model using LSTM neural network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases with low MAPE value.

# =============================================================================
# =============================================================================

# MODEL DEVELOPMENT AND EVALUATION

# Model's MAPE value was around 0.14 when used 64 hidden nodes, 2 hidden layers, 0.2 as drop out rate and 100 epochs.
# Model performed well when hiddens notes were reduced to 16 and drop out rate to be 0.1, MAPE has reduced to 0.11.
# Model performed even better when drop out rate reduced to 0.05 and epochs up to 500 around 0.10
# Model performed the same when second hidden layer was removed. 
# Hence, the parameter set for the model are 
# - no.of hidden layers:1 LSTM layer
# - no of hidden layer nodes:16
# - drop out rate: 0.05
# - epochs:500

# =============================================================================
# =============================================================================

# # CHALLENGES
# - model's accuracy increased just by 0.04 MAPE value.

# =============================================================================
# =============================================================================

# SUGGESTIONS
# - Train using more data
# - Use more relevant features to increase model accuracy 

# =======================================================================























