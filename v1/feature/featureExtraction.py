'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''


import pandas as pd
import numpy as np




class FeatureExtraction(object):
	"""
	
	This class requires DataFrames to be in Time Bar
	This class contains all the functions for feature extraction
	SMA, EMA, WMA, RSI
	
	Also contains feature visualisation functions
	
	"""

	def __init__(self):

		pass


	################################################################################
	######## Helper function to get period of Extraction  ##########################
	################################################################################

	def periodIdx(self, df, TimeColName, TimeThrhld):
		''' compute time bars    
		# args
			df: pd.DataFrame()
			TimeColName: name for time data
			period: threshold value for Time
		# returns
			idx: list of indices where a time based candlestick ends
		'''
		
		t=pd.to_datetime(df[TimeColName])
		idx=[]
		init_time=t.iloc[0]

		for i,x in enumerate(t):
			duration=(x-init_time).total_seconds()
			if(duration>=TimeThrhld):
				idx.append(i)
				init_time=x
				continue
		
		
		return idx


	################################################################################
	######## Helper function to Check dataframe  ###################################
	################################################################################


	def check_df(self,df,cols,dropna=True):
		if dropna == True:
			df = df.dropna()
		if 'DateStop' in df.columns:			
			if 'Date' in df.columns:
				return df
			else:
				# after reset index DateStart is first column
				# rename first columns of dataframe
				tdf = df.reset_index()
				tdf = tdf.rename(columns={tdf.columns[0]:'Date'})
				tdf['Date'] = pd.to_datetime(tdf['Date'])
				# print(tdf.head())

				return tdf

		return df

	################################################################################
	######## Helper function to convert rows list to Dataframe  ####################
	################################################################################

	def rows_to_df(self,rows_list,columns):
		'''
		Utility function
		list of rows of a dataframe to dataframe
		
			# args
				df: time based dataframe (candlestick)
				columns: 
			# returns
				dataframe of given columns and respective rows
		''' 
		
		df = pd.DataFrame(rows_list,columns=columns)
		return df


	################################################################################
	######## Helper function to concate dates_df and ohlc_df  ######################
	################################################################################


	def concat_df_dates_ohlc(self,dates_df,ohlc_df):
		return pd.concat([dates_df,ohlc_df],axis=1)
	


	################################################################################
	######## Helper function to get features according to labels ###################
	################################################################################


	def get_features_acc_labels(self,df1,df2,col_name='Close'):
		"""Get the features of downsampled labels
		args:
			df1: dataframe of labels
			df2: dataframe of feautres
			col_name: name of col to perform extraction
		returns:
			series of col_name prices as per index of labels
		"""
		df2['Date'] = pd.to_datetime(df2['Date'])

		date_list = df1.index

		dt, op, hi, lo, cl = [],[],[],[],[]
		v = []

		if col_name == 'RSI':
			# print("if blk")
			
			rsi = []
			for i in date_list:
				val = df2.loc[df2['Date']==i]
				# print(val.values[0])
				d,o,h,l,c, temp1,temp2, r =val.values[0][0:8]

				dt.append(d)
				op.append(o)
				hi.append(h)
				lo.append(l)
				cl.append(c)
				rsi.append(r)

			temp_dict = {
				'Date':dt,
				'Open':op,
				'High':hi,
				'Low':lo,
				'Close':cl,
				'RSI':rsi
			}
			
			

			df = pd.DataFrame(temp_dict)
			# print(df.columns)
			# print(col_name)
			return df[col_name]

		elif col_name == 'Close' or col_name == 'Open' or col_name =='High' or col_name == 'Low':
			# print("elif blk")
			
			for i in date_list:
				val = df2.loc[df2['Date']==i]
				# print(val.values[0])
				d,o,h,l,c=val.values[0][0:5]
				dt.append(d)
				op.append(o)
				hi.append(h)
				lo.append(l)
				cl.append(c)

			temp_dict = {
				'Date':dt,
				'Open':op,
				'High':hi,
				'Low':lo,
				'Close':cl
			}
			df = pd.DataFrame(temp_dict)
			# print(df)
			return df[col_name]
		
		else:
			# print("else blk")
			# print(col_name)
			
			for i in date_list:
				val = df2[col_name].loc[df2['Date']==i]
				# print(val.values)
				v.append(val.values[0])
			return v

	################################################################################
	######## Helper function to get 5feature cols according to labels ##############
	################################################################################



	def map5features(self,clean_labels,df,step=5,mode='sma'):
		"""
		To get 5 columns of featured mapped according to downsampled labels
		args:
			clean_labels: downsampled labelled dataframe
			df : dataframe containing 5 columns of a feature
		returns:
			dataframe: of 5 columns mapped according to labels
		"""
		l = np.arange(step,step*6,step)
		cols = []

		for i in l:
			cols.append(mode+str(i))
		res_df = pd.DataFrame(columns=cols)
		
		# print(df.columns)
		# print(df.index)
		# skip date column
		for i in cols:
			# print("*************************************************col_name: ",i)
			t = self.get_features_acc_labels(clean_labels,df,col_name=i)
			res_df[i] = t
			# print(len(res_df))


		dates = pd.DataFrame(clean_labels.index)
		res_df = pd.concat([pd.DataFrame(clean_labels.index),res_df],axis=1)
		res_df = res_df.rename(columns={'index':'Date'})
		return res_df


	################################################################################
	######## Helper function to get 5 cols of an extracted features  ###############
	################################################################################

	def get_five_features(self,df,step=5,mode='sma',dropna=True):
		
		"""
		5 feature columns separated by step
		eg : for step = 5 
		calculates 5,10,15,20 sma
		eg : for step = 6 
		calculates 6,12,18,24 sma
		
		args:
			df : pandas DataFrame
			step : period separation between columns
			mode : 'sma', 'ema'
			dropna : To drop dataframe na values
		
		returns:
			5,10,15,20,25 period features
		"""
		
		a1 = np.arange(1,6)

		l = step*a1
		cols = []
		for i in l:
			cols.append(mode+str(i))
		
		# print(cols)
		res_df = pd.DataFrame(columns=cols)
		
		
		if mode == 'sma':
			for x,i in enumerate(l):
				tx = self.simple_moving_avg(df,i)
				close = tx['Close']
				col_name = res_df.columns[x]
				res_df[col_name] = close
				
		elif mode == 'ema':
			for x,i in enumerate(l):
				tx = self.exp_moving_avg(df,i)
				close = tx['Close']
				col_name = res_df.columns[x]
				res_df[col_name] = close
			
		

		dates = pd.DataFrame(data=df['Date'])
		five_df = pd.concat([dates,res_df],axis=1,sort=True)
		
		if dropna:
			return five_df.dropna()
		else:
			return five_df

	################################################################################
	######## Specific function to calculate Simple Moving Average ##################
	################################################################################

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
		


	################################################################################
	######## Specific function to calculate Exp Moving Average ##################
	################################################################################

	def exp_moving_avg(self, df,period,dropna=True):
		''' Exponential moving average
			# args
				df: time based dataframe (candlestick)
				period: each time bar value (in seconds, float)
						EMA for x 'period'
			# returns
				EMA: for time bar
		'''
		# Formula
		"""
		EMA [today] = (Price [today] x K) + (EMA [yesterday] x (1 – K)) 
		
		Where:
		K = 2 ÷(N + 1)
		N = the length of the EMA
		Price [today] = the current closing price
		EMA [yesterday] = the previous EMA value

		EMA [today] = the current EMA value
		"""


		df = self.check_df(df,df.columns)

		# idx = self.periodIdx(df,'Date',period)
		# print(idx)
		# period = idx[1]-idx[0]


		t = pd.DataFrame(df.loc[:,['Open', 'High', 'Low', 'Close']],columns=['Open', 'High', 'Low', 'Close'])
		t = t.ewm(span=period,min_periods=period,adjust=False).mean()
		dt = pd.DataFrame(df.loc[:,['Date']])
		EMA_df = dt.join(t)


		if dropna == True:
			EMA_df = EMA_df.dropna()

		return EMA_df
	################################################################################
	######## Specific function to calculate DEMA ###################################
	################################################################################



	def double_exp_mov_avg(self, df,period,dropna=True):
		''' Double Exponential moving average
			# args
				df: dataframe
				period: each time bar value (in seconds, float)
						DEMA for x 'period'
			# returns
				DEMA: for time bar
		'''
		# Formula
		"""
		DEMA = ( 2 * EMA(n)) - (EMA(n) of EMA(n))
		"""


		df = self.check_df(df,df.columns)

		# idx = self.periodIdx(df,'Date',period)
		# print(idx)
		# period = idx[1]-idx[0]




		EMA_n = self.exp_moving_avg(df,period,dropna=False)
		EMA_of_EMA_n = self.exp_moving_avg(EMA_n,period,dropna=False)
		
		val1 = pd.DataFrame(EMA_n.loc[:,['Open', 'High', 'Low', 'Close']])
		val2 = pd.DataFrame(EMA_of_EMA_n.loc[:,['Open', 'High', 'Low', 'Close']])
		
		val1 = val1.multiply(2.,axis=0)
		
		# corner case due to ema of ema shape is reduced by 1
		# verify that shapes are correct
		# print(val1[:-1:].shape,val2.shape)
		
		# reducing shape
		
		DEMA_val = val1.subtract(val2,axis=0)
		
		DEMA_val = pd.DataFrame(DEMA_val,columns=['Open', 'High', 'Low', 'Close'])
		
		dates_df = pd.DataFrame(df.loc[:,['Date']])
		
		if dropna == True:
			return dates_df.join(DEMA_val).dropna()
		else:
			return dates_df.join(DEMA_val)



	################################################################################
	######## Specific function to calculate TEMA ###################################
	################################################################################


	def triple_exp_moving_avg(self,df,period,dropna=True):
		''' Triple Exponential moving average
			# args
				df: dataframe (candlestick)
				period: each time bar value (float)
						TEMA for x 'period'
			# returns
				TEMA: for df
		'''
		# Formula
		"""
		The Triple Exponential Moving Average (TEMA)
		of time series 't' is:

		* EMA1 = EMA(t,period)

		* EMA2 = EMA(EMA1,period)

		* EMA3 = EMA(EMA2,period))

		* TEMA = 3*EMA1 - 3*EMA2 + EMA3
		"""

		df = self.check_df(df,df.columns)

		# idx = self.periodIdx(df,'Date',period)
		# print(idx)
		# period = idx[1]-idx[0]

		step = period -1

		EMA1 = self.exp_moving_avg(df,period,dropna=False)
		EMA2 = self.exp_moving_avg(EMA1,period,dropna=False)
		EMA3 = self.exp_moving_avg(EMA2,period,dropna=False)

		# corner case
		# shapes of all ema is 1 less than other
		# make all shapes equal

		EMA1 = pd.DataFrame(EMA1.loc[:,['Open', 'High', 'Low', 'Close']])
		EMA2 = pd.DataFrame(EMA2.loc[:,['Open', 'High', 'Low', 'Close']])
		EMA3 = pd.DataFrame(EMA3.loc[:,['Open', 'High', 'Low', 'Close']])

		# verify shapes
		# print(EMA1.shape,EMA2.shape,EMA3.shape)

		TEMA = ((EMA1.multiply(3)).subtract(EMA2.multiply(3))).add(EMA3)

		# join dates and ohlc values dataframe
		dates_df = pd.DataFrame(df.loc[:,['Date']])
		TEMA_df = pd.DataFrame(TEMA,columns=['Open', 'High', 'Low', 'Close'])

		if dropna == True:
			return dates_df.join(TEMA_df).dropna()
		else :
			return dates_df.join(TEMA_df)
	################################################################################
	######## Specific function to calculate Weighted Moving Average ################
	################################################################################

	def weighted_moving_avg(self, df,period):
		''' weighted moving average
		# args
			df: time based dataframe (candlestick)
			period: each time bar value (in seconds, float)
			
		# returns
			WMA: for time bar

		''' 


		df = self.check_df(df,df.columns)

		# idx = self.periodIdx(df,'Date',period)
		# print(idx)
		# period = idx[1]-idx[0]

		
		l = []
			
		step = period - 1 
		n = df.shape[0]
		
		W = np.arange(period)
	
		
		for i in range(0,n):
			date = df['Date'][i]
			
			s1 = pd.Series({
				'Date':date
			})
			
			s2 = df.loc[i:i+step, ['Open', 'High', 'Low', 'Close']].values
			
			s3 = []
			for i,j in zip(W,s2):
				res = i*j
				s3.append(res)
			
			s3 = np.array(s3)

			s3 = s3.sum(axis=0)/(np.sum(W))
			s3 = pd.Series(s3)

			s = s1.append(s3)
			l.append(s.tolist())
		
		WMA = pd.DataFrame(l,columns=['Date','Open', 'High', 'Low', 'Close'])

		dl = WMA['Date'].tolist()
		WMA_dates = pd.DataFrame(dl,columns=['Date'])
		WMA = WMA.iloc[:,1:].shift(step,axis=0)
		
		WMA = WMA_dates.join(WMA)
		WMA = WMA.dropna()
		
		return WMA




	


	
		
	################################################################################
	######## Specific function to calculate MACD ###################################
	################################################################################

		
	def macd_helper(self, dates,df, columns):

		l=[]

		for i in range(0,dates.shape[0]):
			tempi = dates[i]
			tempj = df.iloc[i,:].values
			tempi = list(tempi)
			for j in tempj:
				tempi.append(j)

			l.append(tempi)

		# here 'l' is rows_list

		return self.rows_to_df(l,columns)




	def macd(self,df,p1=12,p2=26,p3=9):
		''' Moving Average Converging Diverging
			# args
				df: time based dataframe (candlestick)
				period: p1,p2,p3

			# returns
				MACD line, Signal line, MACD Histogram
		''' 
		# Formula
		'''
		General Setting for MACD is 12 26 9
		Other values can also be passed

		MACD Line: (12-day EMA - 26-day EMA)

		Signal Line: 9-day EMA of MACD Line

		MACD Histogram: MACD Line - Signal Line

		NOTE: Currently period in seconds
		'''

		df = self.check_df(df,df.columns)

		# idx = self.periodIdx(df,'Date',p1)
		# print(idx)
		# p1 = idx[1]-idx[0]

		# idx = self.periodIdx(df,'Date',p2)
		# print(idx)
		# p2 = idx[1]-idx[0]

		# idx = self.periodIdx(df,'Date',p3)
		# print(idx)
		# p3 = idx[1]-idx[0]

		EMA_12 = self.exp_moving_avg(df,p1,dropna=False)
		EMA_26 = self.exp_moving_avg(df,p2,dropna=False)
		EMA_9 =  self.exp_moving_avg(df,p3,dropna=False)

		"""
		print(EMA_12.shape)
		print(EMA_26.shape)
		print(EMA_9.shape)
		"""

		ohlc_EMA_12 = EMA_12.loc[:,['Open', 'High', 'Low', 'Close']]
		ohlc_EMA_26 = EMA_26.loc[:,['Open', 'High', 'Low', 'Close']]
		ohlc_EMA_9 = EMA_9.loc[:,['Open', 'High', 'Low', 'Close']]

		# MACD Line: (12-period EMA - 26-period EMA)
		MACD_line_res = ohlc_EMA_12.subtract(ohlc_EMA_26,axis=1)

		# Signal Line: 9-period EMA of MACD Line
		Signal_line_res =  ohlc_EMA_9

		# MACD Histogram: MACD Line - Signal Line
		MACD_hist_res = MACD_line_res.subtract(Signal_line_res,axis=1)




		columns=['Date','Open', 'High', 'Low', 'Close']

		# Giving time stamps of EMA_26 to each res
		tstamps = EMA_26.loc[:,['Date']].values
		# print(stamps)

		MACD_line = self.macd_helper(tstamps, MACD_line_res,columns)
		Signal_line = self.macd_helper(tstamps, Signal_line_res,columns)
		MACD_hist = self.macd_helper(tstamps, MACD_hist_res,columns)
		
		MACD_line = MACD_line.dropna()
		Signal_line = Signal_line.dropna()
		MACD_hist = MACD_hist.dropna()
		
		# print(MACD_line.shape,Signal_line.shape,MACD_hist.shape)
		
		return (MACD_line, Signal_line, MACD_hist)





	################################################################################
	######## Specific function to calculate Bollinger Bands ########################
	################################################################################
	"""
	

	"""
	
	def bollinger_bands(self,df, period, std=2,dropna=True):
		
		'''
		Bollinger bands
		args:
			df : TimeBar dataframe
			period (int): period / window of rolling
			std (float): by default stddev value is 2 
		
		returns:
			upbnd,dnbnd (tuple): Bollinger upband and downband dataframe
		'''
		
		
		# Formula
		"""
		Middle Band = N-day simple moving average (SMA)
		Upper Band = N-day SMA + (N-day standard deviation of price x 2)
		Lower Band = N-day SMA – (N-day standard deviation of price x 2)
		"""

		df = self.check_df(df,df.columns)
		
		# idx = self.periodIdx(df,'Date',period)
		# print(idx)
		# period = idx[1]-idx[0]
	

		
		n_sma = self.simple_moving_avg(df,period,dropna=False)
		# n_sd = self.sd_moving(df,period,dropna=False)
		n_sd = df.loc[:,['Open', 'High', 'Low', 'Close']].rolling(window=period).std()
		
		n_sma_values = n_sma.loc[:,['Open', 'High', 'Low', 'Close']]
		n_sd_values = n_sd.loc[:,['Open', 'High', 'Low', 'Close']]
		
		
		up = n_sma_values.add(n_sd_values.multiply(std),axis=0) 
		dn = n_sma_values.subtract(n_sd_values.multiply(std),axis=0) 
		
		up = pd.DataFrame(up,columns=['Open', 'High', 'Low', 'Close'])
		dn = pd.DataFrame(dn,columns=['Open', 'High', 'Low', 'Close'])
		
		
		dates_df = df.loc[:,['Date']]

		upbnd = self.concat_df_dates_ohlc(dates_df,up)
		dnbnd = self.concat_df_dates_ohlc(dates_df,dn)
		
		
		
		
		# nan values will occur due to period of rolling average
		# hence drop and return
		if dropna ==True:
			return (upbnd.dropna(),dnbnd.dropna())
		else:
			return (upbnd,dnbnd)



	################################################################################
	######## Specific function to calculate RSI ####################################
	################################################################################




	def avg_gain_loss(self,values,period):
		s = pd.Series(values).rolling(window=period-1).mean()
		return s
		
		
	def rsi(self,df,col='Price',period=14,dropna=True):

		df = self.check_df(df,df.columns,dropna=False)


		values = df.loc[:,'Price'].diff()
		# print(values)
		
		d = {'g':[0],'l':[0]}
		for i,x in enumerate(values):
			if values[i]>0:
				gain = values[i]
				loss = 0
			else:
				gain = 0
				loss = -1*values[i]
			
			d['g'].append(gain)
			d['l'].append(loss)
			
		
		# calculate avg loss and avg gain

		avg_gain = self.avg_gain_loss(d['g'],period)
		avg_loss = self.avg_gain_loss(d['l'],period)
		
		rs = avg_gain.divide(avg_loss)
		
		# print(rs)
		
		# normalise RS to get RSI
		rsi = 100 - (100/(rs+1))



		res_df = pd.DataFrame({
			'Date':df['Date'],
			'Close':df['Close'],
			'RSI':rsi
			})

		return res_df

	################################################################################
	######## Specific function to calculate VPT ####################################
	################################################################################


	def vpt(self,df,dropna=True):
		"""
		Volume Price Trend 
		
		args:
			df: DataFrame with ohlc volume and price
			dropna: if true drops na while reutrn
		return:
			df: with columns date,close,volume,VPT

		The volume price trend (VPT) indicator helps 
		determine a security’s price direction and strength of price change.
			
		"""
		
		# Formula
		# VPT = VPT['prev'] * (close['today'] - close['prev']) / close['prev']
		
		df = self.check_df(df,df.columns)


		df['vpt_prev'] = df['Volume'].shift(1)
		df['close_prev'] = df['Close'].shift(1)
		
		VPT = df['vpt_prev'].multiply(df['Close'].subtract(df['close_prev'])).div(df['close_prev'])
		df['VPT'] = VPT
		
		dt = df['Date']
		close = df['Close']
		volume = df['Volume']
		vpt = df['VPT']
		
		df = pd.concat(objs=[dt,close,volume,vpt],axis=1)
		
		if dropna:
			return df.dropna()
		return df
		

	################################################################################
	######## Specific function to calculate EMV ####################################
	################################################################################

	def emv(self,df,K=10**6,period=14,dropna=True):
		"""
		Ease of Movement indicator

		args:
			df: dataframe with ohlcv
		return:
			emv : dataframe with date close vol emv
		Ease of movement is a momentum indicator that demonstrates 
		the relationship between the rate of change in an asset’s price and its volume
		"""

		df = self.check_df(df,df.columns)

		H = df['High']
		L = df['Low']
		pH = df['High'].shift(1)
		pL = df['Low'].shift(1)

		vol = df['Volume']


		num = ((H+L)/2.)-((pH+pL)/2)
		den = ((vol)/K) / (H-L)

		emv = num/den

		df['EMV'] = emv
		df['EMV'] = df['EMV'].fillna(0.) # handle divide by zero
		df['EMV'] = df['EMV'].rolling(period).mean()
		df['EMV'] = df['EMV'].fillna(0.) # handle divide by zero
		
		emv = df['EMV']
		dt = df['Date']
		close = df['Close']
		volume = df['Volume']
		
		df = pd.concat(objs=[dt,close,volume,emv],axis=1)



		if dropna:
			return df.dropna()
		return df
	################################################################################
	######## Specific function to calculate Williams %R ############################
	################################################################################


	def willamsr(self,df,period=14):
		"""
		Method for calculating Williams %R
		args:
			df : dataframe
			period : moving window
		returns :
			dtaframe with williams R as feature
		"""
		#  moves between 0 and -100
		# %R = (highest high – closing price) / (highest high – lowest low) x -100
		
		df = self.check_df(df,df.columns)

		dt = []
		cl = []
		williams_r = []
		

		for i in range(df.index[1],df.shape[0]):
			tdf = df.iloc[i:i+14,:]

			# highest high and lowest low
			hh,ll = tdf['High'].max(),tdf['Low'].min()
			
			num = hh-tdf.loc[i,'Close']
			den = hh-ll
			
			try:
				per_R = (num*100*-1)/den
			except Exception as e:
				per_R = 0
			
			
			dt.append(tdf.loc[i,'Date'])
			cl.append(tdf.loc[i,'Close'])
			williams_r.append(per_R)
		
		# print(len(williams_r),len(dt))
		
		res_df = pd.DataFrame({
			'Date':dt,
			'Close':cl,
			'WilliamsR':williams_r
		})
		
		res_df = res_df.fillna(0.)
		return res_df




	################################################################################
	######## Specific function to calculate ROC ####################################
	################################################################################



	def roc(self,df,col_name='Close',period=14,dropna=True):
		"""
		Method to calculate Rate of change
		args:
			df: dataframe ()
			col_name: column name to calculate ROC over
			period: period to look back for ROC
		returns:
			dataframe with roc
		
		
		"""
		df = self.check_df(df,df.columns)
		# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
		tdf = df.copy(deep=True)
		tdf['close_n'] = tdf['Close'].shift(period-1)
		
		roc = (tdf['Close']-tdf['close_n'])*100/tdf['close_n']
		tdf['ROC'] = roc
		tdf= tdf.drop(columns=['close_n'])

		if dropna:
			return tdf.dropna()

		return tdf
	



	################################################################################
	######## Specific function to calculate AD Oscillator ##########################
	################################################################################
	
	def ad_oscillaor(self,df,period):
		"""
		Method for calculating ad_oscillator
		args:
			df : dataframe
		returns:
			dataframe with ad oscillator
		"""
		
		# Formula
		"""
		1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 

		2. Money Flow Volume = Money Flow Multiplier x Volume for the Period

		3. ADL = Previous ADL + Current Period's Money Flow Volume
		"""
		df = self.check_df(df,df.columns)
		tdf = df.copy(deep=True)
		
		# Step 1
		mfm = ((tdf['Close']-tdf['Low']) - (tdf['High']-tdf['Close'])) / (tdf['High']-tdf['Low'])
		mfm = mfm.interpolate(method='pad')
		
		# Step 2
		volume_period = tdf['Volume'].rolling(period).sum()
		# print(volume_period)
		mfv = mfm.multiply(volume_period)
		
		# Step 3
		adl_prev = 0
		adl = [0,]
		for i in mfv:
			if np.isnan(i):
				pass
			else:
				val = adl_prev+i
				adl_prev = val
				adl.append(val)

		adl =pd.Series(adl)
		
		tdf['MFM'] = mfm
		tdf['MFV'] = mfv
		tdf['ADL'] = adl
		tdf['ADL'] = tdf['ADL'].shift(period+2)
		return tdf
		
