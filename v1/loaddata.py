import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import pickle

from load_data.loadData import LoadData
from preprocessing.preProcessData import PreProcessData
from csticks.createcandleStick import createCandleStick
from normalize.norm import Normalizer 

class Loader():
	def __init__(self):
		self.ld_data = LoadData()
		self.pp_data = PreProcessData()
		self.cstk_ob = createCandleStick()
		self.nl_data=Normalizer()

	def create(self,filename,before=True,norm=True,norm_val=100,norm_method='multiply',bar_type='time',threshold=60,flag=1):
		
		pkl_rawData='saved_runs/MAIN_df_'+filename[30:]
		pkl_Bars='saved_runs/cstk_df_'+filename[30:]+'_'+bar_type+'_'+str(threshold)+'_'+str(before)+'_'+str(norm)+'_'+str(norm_val)+'_'+norm_method
		normalized_main=pkl_rawData+'_normalized_main_df'+'_'+str(before)+'_'+str(norm)+'_'+str(norm_val)+'_'+norm_method
		normalized_cstk=pkl_Bars+'_normalized'

		try:
			MAIN_df=pd.read_pickle(pkl_rawData)
		except (OSError, IOError) as e:
			print(e)
			if flag==1:
				MAIN_df=self.ld_data.load_data(filename)
			else:
				MAIN_df = self.ld_data.load_data_dir(filename)
	
			MAIN_df.to_pickle(pkl_rawData)

		MAIN_df.head()

		if before and norm:

			try:
				MAIN_df=pd.read_pickle(normalized_main)
			except (OSError, IOError) as e:
				print(e)
				MAIN_df = self.nl_data.normalize(MAIN_df,norm_val,norm_method,'Open','High','Low','Close','Volume','Price')
				MAIN_df.to_pickle(normalized_main)

		try:
			MAIN_df=pd.read_pickle(pkl_Bars)
		except (OSError, IOError) as e:
			print(e)
			MAIN_df = self.cstk_ob.createBars(MAIN_df,bar_type,threshold,0)
			MAIN_df.to_pickle(pkl_Bars)

		if not before and norm:

			try:
				MAIN_df=pd.read_pickle(normalized_cstk)
			except (OSError, IOError) as e:
				print(e)
				MAIN_df=self.nl_data.normalize(MAIN_df,norm_val,norm_method,'Open','High','Low','Close','Volume','Price')
				MAIN_df.to_pickle(normalized_cstk)

		return MAIN_df
