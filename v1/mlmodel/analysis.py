'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

from mlmodel.validation import Validation
from mlmodel.sequential_bootstrap import sequentialBootstrap
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from string import ascii_letters
import seaborn as sns
import numpy as np
import pandas as pd
import math

class Analyser(object):
    
    """
    This class contains functions for analyzing the dataset and model 
    
    """
    
    def __init__(self):
        pass
    
    
    def check_overfitting(self, clf, X, y, scoring='precision'):
        '''
        check whether model is overfitting the training dataset or not
        #args
            clf: classifier
            X: input features
            y: labels
            scoring: ‘accuracy’,‘balanced_accuracy’, ‘average_precision’, ‘brier_score_loss’, ‘f1’, ‘f1_micro’,
                    ‘f1_macro’, ‘f1_weighted’, ‘f1_samples’, ‘neg_log_loss’, ‘precision’, ‘recall’, ‘roc_auc’
        
        #returns
            print whether overfit or not
        
        '''
        #change validation.py
        val_ob = Validation()
        train_scores, test_scores = val_ob.learning_curve(clf, X, y, scoring=scoring, n_splits=5, display=False)
        
        train_scores = np.array(train_scores)
        test_scores = np.array(test_scores)
        
        train_scores_mean = np.mean(train_scores)
        test_scores_mean = np.mean(test_scores)
        
        #print('Training score : ', train_scores_mean, ' Test score : ', test_scores_mean)
        
        diff = train_scores_mean - test_scores_mean
        if(diff > 0.2):
            print('The model is overfitting the training set')
        else:
            print('The model is unable to overfit')
            
        
    def check_dataset_correlation(self, df):
        '''
        check correlation between features and labels in the given dataset
        #args
            df: dataset
        
        #returns
            plot correlation
        
        '''
        # Compute the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
        
       
    def check_variance_reduction_bagging(self, individual_clf, bagged_clf, X, y, scoring='f1_micro'):
        '''
        check whether variance is reduced by bagging or not
        #args
            individual_clf: single classifier
            bagged_clf: bagged classifier
            X: input features
            y: labels
            scoring: ‘accuracy’,‘balanced_accuracy’, ‘average_precision’, ‘brier_score_loss’, ‘f1’, ‘f1_micro’,
                    ‘f1_macro’, ‘f1_weighted’, ‘f1_samples’, ‘neg_log_loss’, ‘precision’, ‘recall’, ‘roc_auc’
        
        #returns
            reduced variance values for training and testing dataset
        
        '''
        
        train_sizes = [.2, .4, .6, .8, 1.]
        train_sizes_ind, train_scores_ind, validation_scores_ind = learning_curve(estimator = individual_clf, X = X,
                                                   y = y, train_sizes = train_sizes, cv = 5,
                                                   scoring = scoring)
        
        train_scores_ind_var = np.var(train_scores_ind, axis=1)
        validation_scores_ind_var = np.var(validation_scores_ind, axis=1)
       
        train_sizes_bag, train_scores_bag, validation_scores_bag = learning_curve(estimator = bagged_clf, X = X,
                                                   y = y, train_sizes = train_sizes, cv = 5,
                                                   scoring = scoring)
        
        train_scores_bag_var = np.var(train_scores_bag, axis=1)
        validation_scores_bag_var = np.var(validation_scores_bag, axis=1)

        train_scores_var_diffs = train_scores_ind_var - train_scores_bag_var
        validation_scores_var_diffs = validation_scores_ind_var - validation_scores_bag_var
        
        red_train_scores = np.mean(train_scores_var_diffs)
        red_validation_scores = np.mean(validation_scores_var_diffs)
                                                       
        print("Training score variance is reduced by ", red_train_scores)
        print("Validation score variance is reduced by ", red_validation_scores)
                                                       
        return red_train_scores, red_validation_scores

                                                       
    def check_overlap_reduction(self, bagged_clf, indM, avgU):
        '''
        check whether average uniqueness reduces the overlap or not
        #args
            bagged_clf: bagged classifier
            indM: indicator matrix return by getIndMatrix function (binary matrix indicating what 
                  (price) bars influence the label for each observation).
            avgU: mean of the average uniqueness of each observed feature
        
        #returns
            reduced overlapping value
        
        '''
        sb_ob = sequentialBootstrap()

        Avg = []
        for i in range(10):
            indm = indM[bagged_clf.estimators_samples_[i]]
            avgu = sb_ob.getAvgUniqueness(indm)
            avgu = avgu.mean()
            Avg.append(avgu)
            #print(avgu)
       
        red_avgu = np.mean(np.array(Avg))
        
        print("Overlap is reduced by ", red_avgu - avgU)
        return red_avgu - avgu
    
    
    def early_stopping(self, X, y, scoring='precision'):
        '''
        check whether variance is reduced by early stopping or not in case of random forest classifier
        #args
            X: input features
            y: labels
            scoring: ‘accuracy’,‘balanced_accuracy’, ‘average_precision’, ‘brier_score_loss’, ‘f1’, ‘f1_micro’,
                    ‘f1_macro’, ‘f1_weighted’, ‘f1_samples’, ‘neg_log_loss’, ‘precision’, ‘recall’, ‘roc_auc’
            
        #returns
            reduced variance values for training and testing dataset
        
        '''
        
        train_sizes = [.2, .4, .6, .8, 1.]

        clf = RandomForestClassifier(n_estimators=1,criterion='entropy',bootstrap=False,class_weight='balanced_subsample',                                                                                                  min_weight_fraction_leaf=0.05)
        clf = BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=1.,max_features=1.)
    
        train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = clf, X = X,
                                                   y = y, train_sizes = train_sizes, cv = 5,
                                                   scoring = scoring)
        
        train_scores_var = np.var(train_scores, axis=1)
        validation_scores_var = np.var(validation_scores, axis=1)
        
        clf1 = RandomForestClassifier(n_estimators=1,criterion='entropy',bootstrap=False,class_weight='balanced_subsample')
        clf1 = BaggingClassifier(base_estimator=clf1,n_estimators=1000,max_samples=1.,max_features=1.)
    
        train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = clf1, X = X,
                                                   y = y, train_sizes = train_sizes, cv = 5,
                                                   scoring = scoring)
        
        train_scores_var1 = np.var(train_scores, axis=1)
        validation_scores_var1 = np.var(validation_scores, axis=1)
        
        train_scores_var_diffs = train_scores_var - train_scores_var1
        validation_scores_var_diffs = validation_scores_var - validation_scores_var1
        
        red_train_scores = np.mean(train_scores_var_diffs)
        red_validation_scores = np.mean(validation_scores_var_diffs)
        
        print("Training score variance is reduced by ", red_train_scores)
        print("Validation score variance is reduced by ", red_validation_scores)
                                                       
        return red_train_scores, red_validation_scores
    
    
    def Mahalanobis_distance(self, df1, df2):
        '''
        check correlation between two datasets using Mahalanobis distance
        #args
           df1: first dataframe
           df2: second dataframe
            
        #returns
            Mahalanobis_distance
        
        '''
        mean1 = df1.mean(axis=0)
        mean2 = df2.mean(axis=0)
        mean_diff = mean1.subtract(mean2) 
        
        S = df2.cov()
        S_inverse = np.matrix(S).I
        
        res1 = np.dot(mean_diff, S_inverse)
        res2 = np.dot(res1, mean_diff)
        res2 = abs(res2)
        print("Correlation between the dataframes (Mahalanobis distance) is ", math.sqrt(res2))
        
        return math.sqrt(res2)
