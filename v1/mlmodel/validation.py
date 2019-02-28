'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.model_selection import learning_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import check_scoring

from .performanceMetrics import Metrics

class Validation(object):
	
	"""
	This class contains functions to validate a given model
	
	"""

	def __init__(self):
		pass

	################################################################################
	######## Function to plot Learning Curve  ######################################
	################################################################################
	
	def learning_curve(self, estimator, trainX, trainY, scoring=None, n_splits=5, display=True):
		'''
		Function to plot learning curve
			# args
				estimator: machine learning model
				trainX: dataframe of input features for training
				trainY: dataframe of labels for training
				scoring: a string parameter to decide the scorer object.
						 It can be following:
						 ‘accuracy’, ‘average_precision’, ‘brier_score_loss’, ‘f1_micro’, 
						 ‘f1_macro’, ‘f1_weighted’, ‘f1_samples’, ‘neg_log_loss’, ‘precision’, 
						 ‘recall’, ‘roc_auc’
				n_splits: denotes the number of splits used by TimeSeriesSplit strategy or
						  the number of points present on the learning curve plot
			# returns
			   learning curve plot
		''' 
		scorer = check_scoring(estimator, scoring=scoring)
		trainX = np.array(trainX)
		trainY = np.array(trainY)
		
		train_scores = []
		validation_scores = []
		train_sizes = []
		
		tscv = TimeSeriesSplit(n_splits=n_splits)
		for train_index, test_index in tscv.split(trainX):
			X_train, X_test = trainX[train_index], trainX[test_index]
			y_train, y_test = trainY[train_index], trainY[test_index]
			fitted_estimator = estimator.fit(X_train, y_train)
			train_score = scorer(fitted_estimator, X_train, y_train)
			validation_score = scorer(fitted_estimator, X_test, y_test)
			train_scores.append(train_score)
			validation_scores.append(validation_score)
			train_sizes.append(len(train_index))
			if(display == True):
				print("Train score: ", train_score, "Test score: ", validation_score)
		
		plt.figure()
		plt.title('Learning curve')
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		plt.grid()
		plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, validation_scores, 'o-', color="g", label="Cross-validation score")
		plt.legend(loc="best")
		return train_scores, validation_scores
	



	def SFI_custom(self,clf,train_X,train_y,test_X,test_y,drop_close=True,sfi_flag=True):
		"""
		Method to fit and experiment, and check single feature importance
		args:
			train_X: it must have close column
			train_y: train labels
			test_X: it must have close column
			test_y: test labels

			drop_close=True: clf won't fit close while training
			sfi_flag=True: check single feature importance and their sharpe ratios
		returns:
			metric_dict: dictionary of lists d[feature] = [train_score,test_score,sharpe_ratio]
		"""

		met_ob = Metrics()

		if drop_close:
			try:
				trainX = train_X.drop(columns=['close'])
				testX = test_X.drop(columns=['close'])
			except:
				pass
		
		clf_c = clf
		
		print("Train Labels and Count")
		print(pd.Series(train_y).value_counts())
		print("Test Labels and Count")
		print(pd.Series(test_y).value_counts())
		
		print("*** All features Fit and predict score ***")
		
		clf.fit(train_X,train_y)
			
		train_score= clf.score(train_X,train_y)
		test_score= clf.score(test_X,test_y)
		
		y_pred = clf.predict(test_X)
		sr = met_ob.sr_calc(test_X,y_pred)
		
		print(trainX.columns)
		print("train:", train_score, " test:", test_score, " sr:",sr )
		  
			
		if sfi_flag:
			print("*** Single Feature Importance ***")
			metric_dict ={}

			for i in trainX.columns:
				trX= trainX[i].values.reshape((-1,1))
				teX = testX[i].values.reshape((-1,1))
				
				
				
				clf_c.fit(trX,train_y)

				train_score= clf_c.score(trX,train_y)
				test_score= clf_c.score(teX,test_y)

				y_pred = clf_c.predict(teX)
				sr = met_ob.sr_calc(test_X,y_pred)

				metric_dict[i] = [train_score,test_score,sr]
				print("feature:",i ," train:", train_score, " test:", test_score, " sr:",sr )
				print(pd.Series(y_pred).value_counts())
			
			temp = pd.DataFrame(metric_dict).transpose()
			temp.columns = ['train_score','test_score','sharpe_ratio']
			
			
			temp.plot()
			# plt.xticks(rotation=90)

			
			return temp