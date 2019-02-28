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

	def multiply_method(self,df):

		multiplier=100/np.mean(df)
		return df*multiplier
		

	def normalize(self,df,*args):

		for arg in args:
			#df[arg+'_old']=df[arg]
			print(arg)
			if arg=='Date' or arg=='DateStop':
				continue	
			df[arg]=self.multiply_method(df[arg])

		return df