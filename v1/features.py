import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import pickle

# Features
from feature.featureExtraction import FeatureExtraction
from feature.featureExtractionVisual import FVisual
from feature.featureVerifyVisual import VerifyFeature

class FeatureGen():
	def __init__(self):
		self.fe_ob = FeatureExtraction()
		self.fe_vis = FVisual()
		self.fe_verify = VerifyFeature()


	def generate_features(self,filename,cstk_df,feature_list,period_all):

		pkl_features='saved_runs/features_'+filename[30:]+'_'+str(feature_list)+'_'+str(period_all)
		
		try:
			full_fdf = pd.read_pickle(pkl_features)
		
		except (OSError, IOError) as e:
			print(e)
			full_fdf = pd.DataFrame({'close':cstk_df['Close']})
			full_fdf =  full_fdf.reset_index()
			full_fdf = full_fdf.rename(columns = {"DateStart": "Date"})

			for i in range(len(feature_list)):
				for j in range(len(period_all[i])):
        
        # Adding Column to data frame
					col_name_temp = feature_list[i] + '_' + str(period_all[i][j])

					if feature_list[i] == 'sma':
					    df_temp = self.fe_ob.simple_moving_avg(cstk_df,period_all[i][j],dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['Close']})
					    
					if feature_list[i] == 'ema':
					    df_temp = self.fe_ob.exp_moving_avg(cstk_df,period_all[i][j],dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['Close']})
					    
					if feature_list[i] == 'BB':
					    df_temp = self.fe_ob.bollinger_bands(cstk_df,period_all[i][j],dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp[0]['Close']}) 
					    
					if feature_list[i] == 'rsi':
					    df_temp = self.fe_ob.rsi(cstk_df, col='Price', period = period_all[i][j],dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['RSI']}) 
					    
					if feature_list[i] == 'williamsr':
					    df_temp = self.fe_ob.willamsr(cstk_df,period = period_all[i][j])
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['WilliamsR']}) 
					    
					if feature_list[i] == 'roc':
					    df_temp = self.fe_ob.roc(cstk_df,col_name='Close',period = period_all[i][j], dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['ROC']}) 
					    
					if feature_list[i] == 'adl':
					    df_temp = self.fe_ob.ad_oscillaor(cstk_df,period_all[i][j]) # check divisions
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['ADL']})

					if feature_list[i] == 'vpt':
					    df_temp = self.fe_ob.vpt(cstk_df,dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['VPT']}) 

					if feature_list[i] == 'emv':
					    df_temp = self.fe_ob.emv(cstk_df,dropna=False)
					    df_temp2 = pd.DataFrame({col_name_temp:df_temp['EMV']}) 
        
					full_fdf = pd.concat([full_fdf, df_temp2], axis=1)

			full_fdf=full_fdf.set_index('Date')
			full_fdf.to_pickle(pkl_features)
			

		return full_fdf




