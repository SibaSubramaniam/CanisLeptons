'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

# This class contains all the functions to create different type of bars (candlestick) such as Volume bar

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from sklearn.externals import joblib

from .split import Split

class MLClassifier(object):
    def __init__(self):
        pass

    ##############################################################
    ######## Generic function to call ML algo ####################
    ##############################################################
    def ml_classfr(self, X, y, avgU, method, saveModel=False):
        if method == 'LR':
            return lr_classfr(X, y)
        elif method == 'SGD':
            return sgd_classfr(X, y)
        elif method == 'LSTM':
            return model_lstm(X, y)
        elif method == 'RF':
            return model_randomForest(X, y, saveModel, avgU)

    ##############################################################
    ######## Specific function to call ML algo ####################
    ##############################################################

def lr_classfr(X, y):
    ml_model = [ None, float("-inf") ]
    train_X, train_y, valid_X, valid_y = createData_TrainTest(X, y, 0.7) # split training-testing data

    # Create the LogisticRegression object
    clf = LogisticRegression()
    clf = clf.fit(train_X, train_y)
    # Evaluate the learned model on the validation set
    accuracy = clf.score(valid_X, valid_y)
    ml_model = [ clf, accuracy ]
    return ml_model

def sgd_classfr(X, y):
    ml_model = [ None, float("-inf") ]
    train_X, train_y, valid_X, valid_y = createData_TrainTest(X, y, 0.7) # split training-testing data

    # Create the Stochastic GRadient Classifier object
    clf = SGDClassifier()
    clf = clf.fit(train_X, train_y)
    # Evaluate the learned model on the validation set
    accuracy = clf.score(valid_X, valid_y)
    ml_model = [ clf, accuracy ]
    return ml_model


def model_lstm(X, y):
    #LSTM model for time-series data
    #Initialising the LSTM
    lstm_model = Sequential()

    #Adding the first LSTM layer and some Dropout regularisation
    lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (trainX.shape[1], 1)))
    lstm_model.add(Dropout(0.2))

    #Adding a second LSTM layer and some Dropout regularisation
    lstm_model.add(LSTM(units = 50, return_sequences = True))
    lstm_model.add(Dropout(0.2))

    #Adding a third LSTM layer and some Dropout regularisation
    lstm_model.add(LSTM(units = 50, return_sequences = True))
    lstm_model.add(Dropout(0.2))

    #Adding a fourth LSTM layer and some Dropout regularisation
    lstm_model.add(LSTM(units = 50))
    lstm_model.add(Dropout(0.2))

    #Adding the output layer
    lstm_model.add(Dense(units = 1))

    #Compiling the LSTM
    lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    print(lstm_model.summary())

    #Fitting the LSTM to the Training set
    lstm_model.fit(trainX, trainY, epochs = 100, batch_size = 200, verbose = 1)

    #Saving the model
    model.save('lstm_model.h5')

    # Incase the fitting is taking time, we can comment the fit and save, and directly load the model if it is 
    # available in the same folder
    #model = load_model('lstm_model.h5')
    
    scores = model.evaluate(trainX, trainY, verbose=1, batch_size=200)
    return scores
    

def model_randomForest(X, y, saveModel, avgU=1.):
    ml_model = [ None, float("-inf") ]
    rf_split = Split()
    train_X, train_y, valid_X, valid_y = rf_split.train_test_split(X, y, 0.7) # split training-testing data
    
    # Create the LogisticRegression object
    clf = RandomForestClassifier(n_estimators=1,criterion='entropy',bootstrap=False,class_weight='balanced_subsample')
    clf = BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
    clf = clf.fit(train_X, train_y)
    if (saveModel):
        filename = 'trained_randomForest.sav'
        joblib.dump(clf,filename)
    # Evaluate the learned model on the validation set
    accuracy = clf.score(valid_X, valid_y)
    ml_model = [ clf, accuracy ]
    return ml_model
        










