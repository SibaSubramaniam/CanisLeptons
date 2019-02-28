'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''


import pandas as pd
import numpy as np

try:
	from .featureExtraction import FeatureExtraction
except Exception as e:
	pass

try:
	from featureExtraction import FeatureExtraction
except Exception as e:
	pass

class RealTimeExt(object):
	"""
	This class contains methods to extract features from real time data
	"""

	def __init__(self):

		pass



	################################################################################
	######## Helper function to return formatted data frame ########################
	################################################################################

	def process_list(self,l,index):
		"""
		Method to convert real time raw data to specified datatypes
		args:
			(all values are in string)
			l : [list] of values [Time(in milli sec, unix time), Open, High, Low, Close, Price, Volume]
		returns:
			res_l : converted values in required format
		"""
		dt = pd.to_datetime(l[0],unit='ms')
		o,h,l,c,p,v = map(float,l[1:])
		# print(dt,o,h,l,c,p,v)
		
		
		return pd.DataFrame(data={
			'Date':dt,
			'Open':o,
			'High':h,
			'Low':l,
			'Close':c,
			'Price':p,
			'Volume':v
		},index=[index])
		
	

	# sma,ema,dema,tema,rsi,bb_up,bb_dn,close,sma5,sma10,sma15,sma20,sma25,ema5,ema10,ema15,ema20,ema25
	def get_features_real_time(self,df,period=14,ff_min_period=5,step=5):
		"""
		Method to get real time features:
		sma,ema,dema,tema,rsi,bb_up,bb_dn,close,sma5,sma10,sma15,sma20,sma25,ema5,ema10,ema15,ema20,ema25
		
		args:
			df: DataFrame containing values atleast date,o,h,l,c
			period: (int) period value to get sma,ema,dema,tema,rsi,bb_up,bb_dn
			ff_min_period: (int) period value to get five features out of 1 moving average
			step: interval between 2 consecutive 5 feature periods
					- if step = 5 and ff_min_period = 5 then features will be of period 5,10,15,20,25
		
		returns:
			msg : (if all features not available) "Features NOT available"
			features : (if all features available) DataFrame of features
		"""
		fe_ob = FeatureExtraction()

		# ---------------------------------------------------------------
		# Feature Extraction start
		# ---------------------------------------------------------------
		sma = fe_ob.simple_moving_avg(df,period)
		sma = sma['Close']
		
		ema = fe_ob.exp_moving_avg(df,period)
		ema = ema['Close']
		
		dema = fe_ob.double_exp_mov_avg(df,period)
		dema = dema['Close']
		
		tema = fe_ob.triple_exp_moving_avg(df,period)
		tema = tema['Close']
		
		rsi = fe_ob.rsi(df,period=period)
		rsi = rsi['RSI']
		
		bb = fe_ob.bollinger_bands(df,period)
		bb_up = bb[0]['Close']
		bb_dn = bb[1]['Close']
		
		close = df['Close']
		
		
		five_sma_df = fe_ob.get_five_features(df,step=step,mode='sma')
		five_sma_df = five_sma_df.tail(1)
		if (five_sma_df.isnull().values.any() or five_sma_df.empty):
			five_sma_df = five_sma_df[0:0] # assign empty df
		else:
			a = five_sma_df.values.tolist()
			# print(a[0])
			_ ,sma5,sma10,sma15,sma20,sma25 = a[0]
		
		
		
		five_ema_df = fe_ob.get_five_features(df,step=step,mode='ema')
		five_ema_df = five_ema_df.tail(1)
		if (five_ema_df.isnull().values.any() or five_ema_df.empty):
			five_ema_df = five_ema_df[0:0] # assign empty df
		else:
			a = five_ema_df.values.tolist()
			# print(a[0])
			_ ,ema5,ema10,ema15,ema20,ema25 = a[0]

		# ---------------------------------------------------------------
		# Feature Extraction end
		# ---------------------------------------------------------------


		Flag = bool(sma.empty or ema.empty 
					or dema.empty or tema.empty or rsi.empty 
					or bb_up.empty or bb_dn.empty 
					or five_sma_df.empty or five_ema_df.empty )
		
		if Flag:
			print("******* Features NOT available *******")
		else:
			features = np.array([sma.iloc[-1],ema.iloc[-1],dema.iloc[-1],
				  tema.iloc[-1],rsi.iloc[-1],bb_up.iloc[-1],bb_dn.iloc[-1],
				 close.iloc[-1],sma5,sma10,sma15,sma20,sma25,ema5,ema10,ema15,ema20,ema25])
			
			return features

