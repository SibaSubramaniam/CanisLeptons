'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import numpy as np

class Split(object):
    
    """
    This class contains functions to split the dataset given a preset strategy
    
    """

    def __init__(self):
        pass

    ################################################################################
    ######## Functions to split dataset into train and test sets  ####################
    ################################################################################
    
    def train_test_split(self, dataX, dataY, fraction_train_data):
        '''
        Function to split dataset into two sets of train and test dataset without 
        shuffling the order of input features.

            # args
                dataX: dataframe of input features
                dataY: dataframe of labels
                fraction_train_data: fraction of dataset used for training  
            # returns
                trainX, trainY: dataframe of input features and labels for training 
                testX, testY: dataframe of input features and labels for testing
        ''' 
        num_training_data = int (len(dataY)*fraction_train_data)
        idx = np.arange(0 , len(dataY))
        trainX = dataX.iloc[0:num_training_data, :]
        testX = dataX.iloc[num_training_data:, :]
        trainY = dataY.iloc[0:num_training_data]
        testY = dataY.iloc[num_training_data:]
        return trainX, trainY, testX, testY