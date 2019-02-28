'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

class Metrics(object):
	
	"""
	This class contains performance metrics for time-series classification.
	
	"""

	def __init__(self):
		pass

	################################################################################
	######## Function to compute precision, recall and F-score  ####################
	################################################################################
	
	def precision(self, testY, predY, average=None):
		'''
		Function to compute precision score
			# args
			   testY: test set dataframe with true value lables
			   predY: set with predicted value lables
			   average: string, to determine the type of averaging performed on the data
						It can be:
						 None : scores for each class are returned
						
						'micro': Calculate metrics globally by counting the total true 
						 positives, false negatives and false positives.

						'macro': Calculate metrics for each label, and find their 
						 unweighted mean. This does not take label imbalance into account.

						'weighted': Calculate metrics for each label, and find their 
						 average weighted by support (the number of true instances for
						 each label). 
			# returns
			   precision score
		''' 
		return precision_score(testY, predY, average=average)  
	
	def recall(self, testY, predY, average=None):
		'''
		Function to compute recall score
			# args
			   testY: test set dataframe with true value lables
			   predY: set with predicted value lables
			   average: string, to determine the type of averaging performed on the data
						It can be:
						 None : scores for each class are returned
						
						'micro': Calculate metrics globally by counting the total true 
						 positives, false negatives and false positives.

						'macro': Calculate metrics for each label, and find their 
						 unweighted mean. This does not take label imbalance into account.

						'weighted': Calculate metrics for each label, and find their 
						 average weighted by support (the number of true instances for
						 each label). 
			# returns
			   recall score
		''' 
		return recall_score(testY, predY, average=average) 
	
	def f1score(self, testY, predY, average=None):
		'''
		Function to compute f1-score
			# args
			   testY: test set dataframe with true value lables
			   predY: set with predicted value lables
			   average: string, to determine the type of averaging performed on the data
						It can be:
						 None : scores for each class are returned
						
						'micro': Calculate metrics globally by counting the total true 
						 positives, false negatives and false positives.

						'macro': Calculate metrics for each label, and find their 
						 unweighted mean. This does not take label imbalance into account.

						'weighted': Calculate metrics for each label, and find their 
						 average weighted by support (the number of true instances for
						 each label). 
			# returns
			   f1-score
		''' 
		return f1_score(testY, predY, average=average) 
	
	def confusionMatrix(self, testY, predY, labels):
		'''
		Function to compute and plot confusion matrix
			# args
			   testY: test set dataframe with true value lables
			   predY: set with predicted value lables
			   labels: list of class labels
			# returns
			   confusion matrix plot
		''' 
		cm = confusion_matrix(testY, predY, labels)
		
		true_label = []
		predicted_label = []
		for i in labels:
			true_label.append('True '+str(i))
			predicted_label.append('Predicted '+str(i))
			
		cm = pd.DataFrame(cm, true_label, predicted_label)
		print(cm)
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		plt.title('Confusion matrix of the classifier')
		fig.colorbar(cax)
		ax.set_xticklabels([''] + labels)
		ax.set_yticklabels([''] + labels)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		return plt
	
	def precisionRecall_curve(self, testY, predProbY, classes, kind='both'):
		'''
		Function to plot precision recall curve
			# args
			   testY: test set dataframe with true value lables
			   predProbY: set of predicted probabilities for each class
			   classes: list of class labels
			   kind: to define kind of plotting required.
					 It can be:
						 'micro-joint' : plot with micro averaged scores
						 'separate' : separate plots for each class
						 'both' : both of the above plots together
			# returns
			   precision recall curve
		''' 
		precision = dict()
		recall = dict()
		enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
		binarizedY = enc.fit_transform(np.array(testY).reshape(-1, 1))
		n_classes = len(classes)
		for i in range(n_classes):
			precision[i], recall[i], _ = precision_recall_curve(binarizedY[:, i], predProbY[:, i])
	
		precision["micro"], recall["micro"], _ = precision_recall_curve(binarizedY.ravel(), predProbY.ravel())
	
		if kind == 'micro-joint':
			plt = self.precisionRecall_microJoint_curve(precision, recall)
			
		elif (kind == 'separate'):
			plt = self.precisionRecall_separate_curve(precision, recall, classes)
			
		else:
			plt = self.precisionRecall_combined_curve(precision, recall, classes)
		
		return plt
			
	def precisionRecall_microJoint_curve(self, precision, recall):
		'''
		Function to plot precision recall curve with micro averaged scores
			# args
			   precision: list of precision score for each class along with micro averaged score
			   recall: list of recall score for each class along with micro averaged score
			# returns
			   precision recall curve
		''' 
		plt.figure()
		step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
		plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
		plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Micro-averaged over all classes')
		return plt
	
	def precisionRecall_separate_curve(self, precision, recall, classes):
		'''
		Function to plot precision recall curve separately for each class
			# args
			   precision: list of precision score for each class along with micro averaged score
			   recall: list of recall score for each class along with micro averaged score
			   classes: list of class labels
			# returns
			   precision recall curve
		'''  
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		plt.figure(figsize=(10, 8))
		labels = []
		lines = []
		n_classes = len(classes)
		for i, color in zip(range(n_classes), colors):
			l, = plt.plot(recall[i], precision[i], color=color, lw=2)
			lines.append(l)
			labels.append('Precision-recall for class '+ str(i))
		fig = plt.gcf()
		fig.subplots_adjust(bottom=0.25)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Extension of Precision-Recall curve to multi-class')
		plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
		return plt
		
	def precisionRecall_combined_curve(self, precision, recall, classes):
		'''
		Function to plot both of the above precision recall curves.
			# args
			   precision: list of precision score for each class along with micro averaged score
			   recall: list of recall score for each class along with micro averaged score
			   classes: list of class labels
			# returns
			   precision recall curve
		''' 
		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		plt.figure(figsize=(10, 8))
		labels = []
		lines = []
		n_classes = len(classes)
		for i, color in zip(range(n_classes), colors):
			l, = plt.plot(recall[i], precision[i], color=color, lw=2)
			lines.append(l)
			labels.append('Precision-recall for class '+ str(i))
			
		l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
		lines.append(l)
		labels.append('micro-average Precision-recall')
		fig = plt.gcf()
		fig.subplots_adjust(bottom=0.25)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Extension of Precision-Recall curve to multi-class')
		plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
		return plt
			
	
	def return_forward(self,df):
		"""
		args:
			df : dataframe consisting o h l c
		returns:
			one day forward returns in a column
		"""
		tdf = df.copy(deep=True)
		try:
			close = df['Close']
		except Exception as e:
			# raise e
			close = df['close']
		
		f_close = close.shift(-1)
		tdf['Return'] = ((f_close/close)-1).dropna()

		return tdf

	def sharpe_ratio(self,df,risk=0,dropna=True):
		"""
		args:
			df : dataframe with at least close,labels column
		returns:
			sharpe ratio as a column in dataframe
		"""


		# Formula
		# Sharpe ratio = (Return (t+1)* Predicted_Labels(t) -  C)/ std(Return (t+1)* Predicted_Labels(t))
		tdf = df.copy(deep=True)
		df = self.return_forward(df)
		print("std = ", (df['Return']*df['label']).std() )

		# Formula
		# Sharpe ratio = (Return (t+1)* Predicted_Labels(t) -  C)/ std(Return (t+1)* Predicted_Labels(t))
		tdf = df.copy(deep=True)
		df = self.return_forward(df)


		tdf['Return'] = df['Return']
		# tdf['sharpe_ratio'] = (((df['Return'])*df['label']).mean() - risk)/(df['Return']*df['label']).std()
		sr = ((df['Return']*df['label']).mean() - risk)/(df['Return']*df['label']).std()
		profit = (df['Return']*df['label']).mean() - risk
		ret = (df['Return']*df['label']).std()
		if dropna:
			tdf =tdf.dropna()
			return tdf,sr,profit,ret
		else:
			return tdf,sr,profit,ret



	def sr_calc(self,test_X,y_pred,risk=0,dropna=True):
		"""
		calculate sharpe ratio for test_X and predictions  
		"""
		test_X = test_X.copy()
		test_X['label'] = pd.Series(y_pred,index = test_X.index)
		r = self.sharpe_ratio(test_X,risk=0,dropna=True)
		return r[1]