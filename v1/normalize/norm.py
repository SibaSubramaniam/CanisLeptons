'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import numpy as np

class Normalizer(object):
	def __init__(self):
		pass

	def multiply_method(self,df,const=100):

		multiplier=const/np.mean(df['Close'])

		df['Open']=df['Open']*multiplier
		df['Low']=df['Low']*multiplier
		df['High']=df['High']*multiplier
		df['Close']=df['Close']*multiplier
		df['Price']=df['Price']*multiplier

		multiplier=const/np.mean(df['Volume'])
		df['Volume']=df['Volume']*multiplier

		return df


	def min_max_method(self,df_,period=300,*args):
    
		df=df_[list(args)]
		start=0
		for i in range(period,len(df),period):
		    temp=df[start:i]
		    max_=temp.max()
		    min_=temp.min()
		    df[start:i]=(temp-min_)/(max_-min_)
		    start=i

		print('loop ended')
		temp=df[start:]
		max_=temp.max()
		min_=temp.min()
		df[start:]=(temp-min_)/(max_-min_)

		df_[list(args)]=df
		return df_



	def normalize(self,df,value,method='multiply',*args,):

		if method=='multiply':
			df=self.multiply_method(df,value)
			return df

		if method =='min_max':
			df=self.min_max_method(df,value,*args)
			return df
