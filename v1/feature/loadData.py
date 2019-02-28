'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import glob
import os

class LoadData(object):
	def __init__(self):
		pass

	def load_data_dir(self,foldername):

		df=pd.DataFrame()

		for filename in sorted(glob.glob('../../'+foldername+'/*.csv')):
			temp_df=pd.read_csv(filename)
			df=df.append(temp_df,ignore_index = True)

		return df

