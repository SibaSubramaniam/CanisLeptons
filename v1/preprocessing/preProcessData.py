'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

# This class contains all the functions to create different type of bars (candlestick) such as Volume bar

import pandas as pd


class PreProcessData(object):
	def __init__(self):
		pass

	##############################################################
	######## Drop nan etc  #######################################
	##############################################################
	def removeNaN(self, df):
		''' Remove index with NaN from dataframe
		# args
			df: time based dataframe (candlestick)
		'''  
		df = df.dropna()

		return df


	def fill_missing_vals_col(self,col):
		'''
		Fill missing values in column
		'''

	def fill_missing_vals_df(self,df):
		'''
		Fill missing values in dataframe

		'''
	def drop_duplicate(self,df):

		df=df.drop_duplicates(subset='Date')
		return df

	def change_index(self,df):
		'''
		Convert index of dataframe to DatetimeIndex
		'''
		df=df.set_index('DateStart')
		df.index=pd.DatetimeIndex(df.index)

		return df