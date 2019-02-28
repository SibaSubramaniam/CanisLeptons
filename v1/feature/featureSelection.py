'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''



import numpy as np
import pandas as pd




class FeatureSelection(object):
	"""
	docstring for FeatureSelection
	contains methods for selecting features	
	"""

	def __init__(self):
		pass

	################################################################################
	######## Specofic function to get Correlation matrix  ##########################
	################################################################################

	def get_corr_matrix_select(self,df,remove_highest_col=True):
		"""
		Returns second highest values of correlation between columns
		args:
			df : dataframe whose correlation matrix is to be found.
			remove_highest_col : if true removes the column with highest correlation
		returns:
			tuple: dataframe of (rows,cols,corr values) and dataframe with dropped col
			
		"""
		corr = df.corr(min_periods=14)
		# corr= pd.DataFrame(corr.values,columns=corr.columns)

		max2 = []
		for i in corr.columns:
			max2.append(np.sort(corr[i].values)[-2])
		
		max2_index = []
		for i,j,k in zip(max2,corr.columns,corr.index):
			max2_index.append((corr.loc[corr[j]==i].index[0],j,i))
		
		max2_df = pd.DataFrame(max2_index,columns=['row','col','value'])
		max2_df = max2_df.sort_values(by='value',ascending=False)
		
		# print(max2_df['col'][0])
		col_name = max2_df['col'][0]
		
		# remove column with max correlation
		if remove_highest_col:
			tdf = df.drop(col_name,axis=1)
		else:
			tdf = df
		return (max2_df,tdf)



	################################################################################
	######## Helper function to get Eigen Values and Vectors #######################
	################################################################################


	def get_eVec(self,dot,varThres):
		"""
		Calculate eigen Vectors
		args:
			dot: dot product array of standardized features and it's transpose
			varThres: Variance threshold 
		returns:
			returns: eVal,eVec (i.e eigen value and eigen vector)

		"""

		# compute eVec from dot prod matrix, reduce dimension
		eVal,eVec = np.linalg.eigh(dot)
		idx = eVal.argsort()[::-1] # arguments for sorting eVal desc
		eVal,eVec = eVal[idx],eVec[:,idx]
		
		#2) only positive eVals
		eVal = pd.Series(eVal,index = ['PC_'+str(i+1) for i in range(eVal.shape[0])])
		eVec = pd.DataFrame(eVec,index = dot.index,columns = eVal.index)
		eVec = eVec.loc[:,eVal.index]
		
		#3) reduce dimension, form PCs
		cumVar = eVal.cumsum()/eVal.sum()
		dim = cumVar.values.searchsorted(varThres)
		eVal,eVec = eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
		return eVal,eVec



	
	################################################################################
	######## Specific function to get Orthogonal Features ##########################
	################################################################################

	def orthoFeats(self,dfX,varThres = .95):
		"""
		Method to calculate orthogonal features
		args:
			dfX	: features dataframe 
			varThres = .95 : Variance threshold 
			(computes smallest number of orthogonal features with variance of atleast 0.95)

		returns:
			dfP : orthogonal Features 

		"""

		# Given a dataframe dfX of features, compute orthofeatures dfP
		dfZ = dfX.sub(dfX.mean(),axis = 1).div(dfX.std(),axis = 1) # standardize
		dot = pd.DataFrame(np.dot(dfZ.T,dfZ),index = dfX.columns,columns = dfX.columns)
		eVal,eVec = self.get_eVec(dot,varThres)
		dfP = np.dot(dfZ,eVec)

		
		ex_var = np.var(eVec,axis=0)
		ex_var_ratio = ex_var/np.sum(ex_var)
		ex_var_ratio = list(ex_var_ratio)
		print("Explained Variance Ratio: ",ex_var_ratio)
		print(eVal)
		return dfP
