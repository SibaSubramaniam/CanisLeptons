'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import pickle

from loaddata import Loader
from labeler import LabelGen
from features import FeatureGen

class DataGen():
	''
	def __init__(self):
		self.load=Loader()
		self.label=LabelGen()
		self.feature=FeatureGen()


	def create_data_helper(self,filename,feature_list,period_all,before=True,norm=True,norm_val=100,norm_method='multiply',bar_type='time',threshold=60,sampling=False,volatility_threshold=10,v_bars_duration=1,barrier_conf=[1,1],min_return=0,risk=0,sign_label=True,flag=1):

		MAIN_df=self.load.create(filename,before,norm,norm_val,norm_method,bar_type,threshold,flag)
		labels_df=self.label.generate_labels(filename+'_'+str(before)+'_'+str(norm)+'_'+str(norm_val)+'_'+norm_method,MAIN_df,sampling,volatility_threshold,
							v_bars_duration,barrier_conf,min_return,risk,sign_label)

		features_df=self.feature.generate_features(filename+'_'+str(before)+'_'+str(norm)+'_'+str(norm_val)+'_'+norm_method,MAIN_df,feature_list,period_all)

		#non_zero_labels=labels_df[labels_df['label']!=0.0]
		print('Labels: ', labels_df.label.value_counts())

		# Adding labels to features

		#print(MAIN_df.head())
		#print(labels_df.head())
		#print(features_df.head())

		a=features_df.index.searchsorted(labels_df.index)
		full_df=features_df.iloc[a].dropna()

		full_df['label']=labels_df.label
		#print(full_df.head())

		MAIN_df['date']=MAIN_df.index
		labels_df['date']=labels_df.index
		full_df['date']=full_df.index

		return MAIN_df,labels_df,full_df

	def split(self,df,split_ratio):
		df=df.dropna()
		split_point=int(split_ratio*len(df))
		train=df[:split_point]
		test=df[split_point:]
		return train,test

	def preprocess(self,df):
		X = df.iloc[:, :-2]
		y = df.iloc[:,-2]
		return X,y

	def create_data(self,folder_name,feature_list,period_all,before=True,norm=True,norm_val=100,norm_method='multiply',bar_type='time',threshold=60,sampling=False,volatility_threshold=10,v_bars_duration=1,barrier_conf=[1,1],min_return=0,risk=0,sign_label=True,split=0.7):

		df=pd.DataFrame()
		l_df=pd.DataFrame()
		final_df=pd.DataFrame()
		train=pd.DataFrame()
		test=pd.DataFrame()


		if not glob.glob('../'+folder_name+'/*.csv'):
			for filename in sorted(glob.glob('../'+folder_name+'/*')):
				
				MAIN_df,labels_df,full_df=self.create_data_helper(filename,feature_list,period_all,before,norm,norm_val,norm_method,bar_type,threshold,sampling,volatility_threshold,v_bars_duration,barrier_conf,min_return,risk,sign_label,1)

				df=df.append(MAIN_df,ignore_index = True)
				l_df=l_df.append(labels_df,ignore_index = True)
				final_df=final_df.append(full_df,ignore_index = True)
				
				train_,test_=self.split(full_df,split)
				train=train.append(train_,ignore_index=True)
				test=test.append(test_,ignore_index=True)


		else:
			MAIN_df,labels_df,full_df=self.create_data_helper(folder_name,feature_list,period_all,before,norm,norm_val,norm_method,bar_type,threshold,sampling,volatility_threshold,v_bars_duration,barrier_conf,min_return,risk,sign_label,0)
			df=df.append(MAIN_df,ignore_index = True)
			l_df=l_df.append(labels_df,ignore_index = True)
			final_df=final_df.append(full_df,ignore_index = True)

			train_,test_=self.split(full_df,split)
			train=train.append(train_,ignore_index=True)
			test=test.append(test_,ignore_index=True)

		return df,l_df,final_df,train,test

