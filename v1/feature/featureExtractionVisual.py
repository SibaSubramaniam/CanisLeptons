'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''


import pandas as pd
import numpy as np

# plotly modules
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import IFrame
import plotly.graph_objs as go
import plotly.plotly as py
init_notebook_mode(connected=True)


import json
import requests
from requests.auth import HTTPBasicAuth


from plotly import tools



class FVisual(object):
	"""
	This class requires DataFrames to be in Time Bar
	This class contains all the functions for visualising extracted Features
	SMA, EMA, WMA, MACD, Bollinger Bands
	
	
	
	"""

	def __init__(self):
		"""
		username = 'ssrr' # Replace with YOUR USERNAME
		api_key = 'K7WvR2Whf37Skra8mSwm' # Replace with YOUR API KEY

		auth = HTTPBasicAuth(username, api_key)
		headers = {'Plotly-Client-Platform': 'python'}
		
		init_notebook_mode(connected=True)
		plotly.tools.set_credentials_file(username=username, api_key=api_key)
		"""
		pass

	################################################################################
	######## Helper function to Check dataframe  ###################################
	################################################################################


	def check_df(self,df,cols,dropna=True):
		if dropna == True:
			df = df.dropna()

		if cols[1] == 'DateStop':
			# rename column DateStop to Date
			df.rename(columns={'DateStop':'Date'},inplace=True)

		return df


	#################################################################
	######## Helper function for visualising bollinger bands ########
	#################################################################


	def simple_moving_avg(self, df,period,dropna=True):
		''' Simple moving average
		# args
			df: time based dataframe (candlestick)
			period: each time bar value (in seconds, float)
			
		# returns
			SMA: for time bar
		''' 

		df = self.check_df(df,df.columns)

		# idx = self.periodIdx(df,'Date',period)
		# print(idx)
		# period = idx[1]-idx[0]

		date_df = pd.DataFrame(df.loc[:,'Date'])
		
		
		temp_df = pd.DataFrame(df.loc[:,['Open', 'High', 'Low', 'Close']])
		temp_df = temp_df.rolling(window=period).mean()
		
		# print(date_df.join(temp_df))

		SMA = date_df.join(temp_df)
		if dropna == True:
			SMA = SMA.dropna()
		return SMA


	#################################################################
	######## Specific function to visualise Moving AVGs #############
	#################################################################


	

	def features_visual(self, Date_df,ohlc_one_col,dfTimeBar,SMA_df,WMA_df,EMA_df):
		'''
		Visualisation of features:
		args:
			Date_df (str) : Date | Date
			ohlc_one_col (str) : name of columns one of o,h,l,c
		returns:
			plot of Moving averages and one col 
		'''
		
		
		# df = self.check_df(df,df.columns)
		
		
		trace2 = go.Scatter(
			x=SMA_df[Date_df],
			y=SMA_df[ohlc_one_col],
			mode='lines+text',
			name = 'SMA '+ ohlc_one_col +' points'

		)
		trace3 = go.Scatter(
			x=WMA_df[Date_df],
			y=WMA_df[ohlc_one_col],
			mode='lines+text',
			name = 'WMA '+ ohlc_one_col +'points'
		)
		trace4 = go.Scatter(
			x=EMA_df[Date_df],
			y=EMA_df[ohlc_one_col],
			mode='lines+text',
			name = 'EMA '+ ohlc_one_col +'points'
		)

		trace1 = go.Scatter(
			x=dfTimeBar[Date_df],
			y=dfTimeBar[ohlc_one_col],
			mode='lines+markers',
			name = ohlc_one_col +'points'
		)

		data = [trace2,trace3,trace4,trace1]
		return plotly.offline.iplot(data,filename='MovingAvg.png')





	#################################################################
	######## Specific function to visualise MACD ####################
	#################################################################

	def macd_visual(self,MACD_hist_df,MACD_line_df,Signal_line_df, df, ema_p1=12, ema_p2=26, ema_p3=9, one_ohlc_col='Close'):
		from featureExtraction import FeatureExtraction

		'''
		Visualisation of features:
		args:
			MACD_line_df : Dataframe whose prices are converted to MACD
			Signal_line_df : Dataframe containing signal of respective MACD line
			df : original dataframe unprocessed containing O,H,L,C 
			ohlc_one_col (str) : name of columns one of Open,High,Low,Close
		returns:
			plot of MACD line and Signal line
		'''
		fe_ob = FeatureExtraction()

		ema12_df = fe_ob.exp_moving_avg(df,period=ema_p1)
		ema26_df = fe_ob.exp_moving_avg(df,period=ema_p2)
		# ema9_df = fe_ob.exp_moving_avg(df,period=ema_p3)
		
		
		dat = {'ema12' : ema12_df.loc[:,one_ohlc_col],
			   'ema26' : ema26_df.loc[:,one_ohlc_col],
			   'close' : df.loc[:,one_ohlc_col],
			   'signal' : Signal_line_df.loc[:,one_ohlc_col],
			   'macd_line': MACD_line_df.loc[:,one_ohlc_col],
			   'macd_hist':MACD_hist_df.loc[:,one_ohlc_col]
			  }
		
		#
		
		trace_ema12 = go.Scatter(
				x = ema12_df['Date'],
				y = dat['ema12'],
				name='EMA 12'
			)

		trace_ema26 = go.Scatter(
			x = ema26_df['Date'],
			y = dat['ema26'],
			name='EMA 26'
		)


		close_Price = go.Scatter(
			x=df['Date'],
			y=dat['close'],
			name = one_ohlc_col + ' Price',
			mode = 'lines+markers',
			line = dict(width = 2,color = 'rgb(255, 0, 0)')
		)

		# EMA 9
		signal_line = go.Scatter(
			x=Signal_line_df['Date'],
			y=dat['signal'],
			name = 'Signal line',
			line = dict(width = 2,color = 'rgb(102, 0, 102)')
		)

		# 12 EMA - 26 EMA
		macd_line = go.Scatter(
			x=MACD_line_df['Date'],
			y=dat['macd_line'],
			mode='lines',
			name = 'MACD line',
			line = dict(width = 2,color = 'rgb(0, 102, 0)')
		)
		
		# MACD line - Signal line
		macd_hist = go.Scatter(
			x=MACD_line_df['Date'],
			y=dat['macd_hist'],
			name = 'MACD hist',
			line = dict(width = 2,color = 'rgb(255, 148, 0)')
		)
		
		
		data1 = [trace_ema12,trace_ema26,close_Price]
		f1 = go.Figure(data=data1)
		
		data2 = [macd_line,macd_hist]
		
		f2 = tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
		
		f2.append_trace(macd_line, 1, 1)
		f2.append_trace(signal_line, 2,1)
		f2.append_trace(macd_hist, 3, 1)
		
		
		f2['layout'].update(barmode ='relative')
		
		return f1,f2
		




	#################################################################
	###### Specific function to visualise bollinger bands ###########
	#################################################################
	   


	def bb_bands_visual(self, df, df_upbnd, df_dnbnd, period, one_ohlc_col):
		'''
		Visualisation of features:
		args:
			df : Dataframe 
			df_upbnd : Dataframe  Upperband of Bollinger bands
			df_dnbnd : Dataframe  Lowerband of Bollinger bands
			period : period/window of moving average
			one_ohlc_col (str) : name of columns one of Open,High,Low,Close
		returns:
			plot :plot bollinger bands for given data
		'''
		# df = self.check_df(df,df.columns)
		# df_upbnd = self.check_df(df_upbnd,df_upbnd.columns)
		# df_dnbnd = self.check_df(df_dnbnd,df_dnbnd.columns)

		up = df_upbnd[one_ohlc_col]
		dn = df_dnbnd[one_ohlc_col]
		
		trace1 = go.Scatter(
			x=df_upbnd['Date'],
			y=df_upbnd[one_ohlc_col],
			mode='lines',
			name = 'Upband'

		)

		trace2 = go.Scatter(
			x=df_dnbnd['Date'],
			y=df_dnbnd[one_ohlc_col],
			name = 'Downband'
		)


		sma_df = self.simple_moving_avg(df,period)

		trace3 = go.Scatter(
			x =sma_df['Date'],
			y = sma_df['Close'],
			name = 'SMA'

		)
		
		trace4 = go.Scatter(
			x =df['Date'],
			y =df[one_ohlc_col],
			mode = 'lines',
			name = one_ohlc_col+' Price'
		)
		
		"""
		trace5 = go.Ohlc(
			x =df['Date'],
			open =df['Open'],
			high =df['High'],
			low =df['Low'],
			close =df['Close']
		)"""
		
		data = [trace1,trace2,trace3,trace4]
		return plotly.offline.iplot(data,filename='BollingerBands.png')
		


	#################################################################
	###### Specific function to visualise DEMA ######################
	#################################################################


	def DEMA_visual(self,df,DEMA_df,one_ohlc_col,period):
	
		# df = self.check_df(df,df.columns)

		trace1 = go.Scatter(
			x=df['Date'],
			y=df[one_ohlc_col],
			name=one_ohlc_col+' Price',
			mode='lines'
						   )
		trace2 = go.Scatter(
			x = DEMA_df['Date'],
			y = DEMA_df[one_ohlc_col],
			name='DEMA',
			mode='lines'
		)

		data = [trace1,trace2]
		return plotly.offline.iplot(data,filename='DEMA.png')



	#################################################################
	###### Specific function to visualise TEMA ######################
	#################################################################


	def TEMA_visual(self,df,TEMA_df,one_ohlc_col,period):
	
		# df = self.check_df(df,df.columns)

		trace1 = go.Scatter(
			x=df['Date'],
			y=df[one_ohlc_col],
			name=one_ohlc_col+' Price',
			mode='lines'
						   )
		trace2 = go.Scatter(
			x = TEMA_df['Date'],
			y = TEMA_df[one_ohlc_col],
			name='TEMA',
			mode='lines'
		)
		
		data = [trace1,trace2]
		return plotly.offline.iplot(data,filename='TEMA.png')


	#################################################################
	###### Specific function to visualise RSI  ######################
	#################################################################

	def rsi_visual(self,rsi_df,df):
		

		trace0 = go.Scatter(
			x=df['Date'],
			y=df['Close'],
			mode='lines+markers',
			name = 'Close'
		)
		
		trace2 = go.Scatter(
			x=rsi_df['Date'],
			y=rsi_df['RSI'],
			mode='lines+markers',
			name = 'RSI'
			
			
		)

		fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True,shared_yaxes=True)
		
		fig.append_trace(trace0, 1, 1)
		fig.append_trace(trace2, 2, 1)
		
		

		fig['layout'].update(height=600, width=800,
							 title='RSI plot')
		return plotly.offline.iplot(fig)

	#################################################################
	###### Specific function to visualise VPT  ######################
	#################################################################


	def vpt_visual(self,df):
		"""
		Fucntion to visualise VPT
		"""
		
		close = go.Scatter(
			x = df['Date'],
			y = df['Close'],
			mode = 'lines',
			name = 'Close'
			
		)
		
		vpt = go.Scatter(
			x = df['Date'],
			y = df['VPT'],
			mode = 'lines',
			name = 'VPT'
		)
		
		fig = tools.make_subplots(rows=2, cols=1,shared_xaxes=True)

		fig.append_trace(close, 1, 1)
		fig.append_trace(vpt, 2, 1)
		
		fig['layout'].update(height=600, width=800, title='VPT')
		return iplot(fig)



	#################################################################
	###### Specific function to visualise EMV  ######################
	#################################################################

	def emv_visual(self,df):
		"""
		Fucntion to visualise EMV
		"""

		close = go.Scatter(
			x = df['Date'],
			y = df['Close'],
			mode = 'lines',
			name = 'Close'

		)

		emv = go.Scatter(
			x = df['Date'],
			y = df['EMV'],
			mode = 'lines',
			name = 'EMV'
		)

		fig = tools.make_subplots(rows=2, cols=1,shared_xaxes=True)

		fig.append_trace(close, 1, 1)
		fig.append_trace(emv, 2, 1)

		fig['layout'].update(height=600, width=800, title='EMV')
		return iplot(fig)


	#################################################################
	###### Specific function to visualise Williams %R  ##############
	#################################################################


	def williams_r_visual(self,df):
		"""
		Plot Williams %R
		args:
			df: dataframe
			with keys(Date,Close,WilliamsR)
		returns:
			plot of close price and williamsR 
		"""
		
		dt = df['Date']
		cl = df['Close']
		wr = df['WilliamsR']
		
		
		cl_p = go.Scatter(
			x=dt,
			y=cl,
			mode='lines',
			name='Close'
		)
		
		wr_p = go.Scatter(
			x=dt,
			y=wr,
			mode='lines',
			name = "Williams %R"
		)
		
		fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
		
		fig.append_trace(cl_p,1,1)
		fig.append_trace(wr_p,2,1)
		
		fig['layout'].update(height=600, width=800,
							 title='Williams R plot')
		return iplot(fig)



	#################################################################
	###### Specific function to visualise ROC  ######################
	#################################################################
		
	def roc_visual(self,df):
		"""
		Plot ROC
		args:
			df: dataframe
			with keys atleast (Date,Close,ROC)
		returns:
			plot of close price and ROC 
		"""

		dt = df['Date']
		cl = df['Close']
		roc = df['ROC']


		cl_p = go.Scatter(
			x=dt,
			y=cl,
			mode='lines',
			name='Close'
		)

		roc_p = go.Scatter(
			x=dt,
			y=roc,
			mode='lines',
			name = "ROC"
		)

		fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)

		fig.append_trace(cl_p,1,1)
		fig.append_trace(roc_p,2,1)

		fig['layout'].update(height=600, width=800,
							 title='ROC plot')
		return iplot(fig)




	#################################################################
	###### Specific function to visualise AD oscillator #############
	#################################################################
		
	def ad_osc_visual(self,df):
		"""
		Plot AD oscillator
		args:
			df: dataframe
			with keys atleast (Date,Close,Volume,ROC)
		returns:
			plot of close price and ROC 
		"""
		

		dt = df['Date']
		cl = df['Close']
		mfm = df['MFM']
		adl = df['ADL']
		vol = df['Volume']


		cl_p = go.Ohlc(
			x=df['Date'],
			open=df['Open'],
			high=df['High'],
			low=df['Low'],
			close=df['Close'],
			name = 'OHLC'
		)

		mfm_p = go.Scatter(
			x=dt,
			y=mfm,
			mode='lines',
			name = "MFM"
		)
		
		adl_p = go.Scatter(
			x=dt,
			y=adl,
			mode='lines',
			name = "ADL"
		)
		
		vol_p = go.Bar(
			x=dt,
			y=vol,
			name = "Volume"
		)

		fig = tools.make_subplots(rows=4,cols=1,shared_xaxes=True)

		fig.append_trace(cl_p,1,1)
		fig.append_trace(mfm_p,2,1)
		fig.append_trace(adl_p,3,1)
		fig.append_trace(vol_p,4,1)
		

		fig['layout'].update(height=600, width=800,
							 title='ADL plot')
		return iplot(fig)