![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

# Prediction_Covid19_Case_using_Deep_Learning_LSTM
## Created a deep learning model using LSTM neural network to predict new cases in Malaysia using the past 30 days of number of cases

## PROBLEM STATEMENT
### To create a deep learning model using LSTM neural network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases with low MAPE value.

## EXPLORATORY DATA ANALYSIS

### Questions:
     1. What kind of data are we dealing with?
        - The data has 31 variables including date
        - The target variable for this dataset is 'cases_new'.
 
     2. Do we have missing values?
       - The cases_new variable has 6 '?' and blank spaces.
       - In which coerced the data into numeric which this values is considered as NaNs 

     3. Do we have duplicated datas?
        - None 
        
     4. Do we have extreme values?
       - Yes, but not to worry due to the nature of the dataset.
       
     5. How to choose the features to make the best out of the provided data?
       - Only selected as 'cases_new' as our variable.
       
       
![covid_case_visual](https://user-images.githubusercontent.com/105897390/175527218-4c71a090-2b8a-4e5d-9959-28a4f07d74c5.png)

*This is the trend of covid-19 cases in Malaysia from 25th January 2020 till 4th December 2021.*


### MODEL DEVELOPMENT AND EVALUATION

 Model's MAPE value was around 0.14 when used 64 hidden nodes, 2 hidden layers, 0.2 as drop out rate and 100 epochs.
 Model performed well when hiddens notes were reduced to 16 and drop out rate to be 0.1, MAPE has reduced to 0.11.
 Model performed even better when drop out rate reduced to 0.05 and epochs up to 500 around 0.10
 Model performed the same when second hidden layer was removed. 
 Hence, the parameter set for the model are 
 - no.of hidden layers:1 LSTM layer
 - no of hidden layer nodes:16
 - drop out rate: 0.05
 - epochs:500
 - 
![model_architecture](https://user-images.githubusercontent.com/105897390/175527466-42e95f65-0a1c-4702-a172-1c3f8fd09be3.png)

*This is architecture of our developed model* 

![Actual_vs_Predicted](https://user-images.githubusercontent.com/105897390/175527451-a1a3af53-d01a-4d0f-bf02-d0b64c3be9b0.png)

*The graph depicts the predicted and actual number of Covid-19 cases from the model that we have developed.*

![MAPE_value](https://user-images.githubusercontent.com/105897390/175527801-98150cff-db87-43cc-abeb-600efab81b3c.png)

*The image of the console potrays the MAPE value obtained while tested on test datasetb by using the developed model*


![tensorboard_loss_mape](https://user-images.githubusercontent.com/105897390/175528194-72403b84-bdc4-4cbd-b82f-bf9369518843.png)
*This tensorboard graph shows the comparison of a few models's loss and MSE value against epoch. We can see that the higher the value of epoch, the better the value of both value of loss and MAPE.*


![tensorboard_final_model](https://user-images.githubusercontent.com/105897390/175528079-a3a6f013-fab2-4b9b-adb3-5b9af43beb09.png)

*This tensorboard depicts the flunctuation of both loss and MAPE value of our finalized model's applied on train dataset.*

### CHALLENGES
 - model's accuracy increased just by 0.04 MAPE value.


### SUGGESTIONS
 - Train using more data
 - Use more relevant features to increase model accuracy 
