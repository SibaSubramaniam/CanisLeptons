'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

# This class contains all the functions to create different type of bars (candlestick) such as Volume bar

import pandas as pd


class createCandleStick(object):
	def __init__(self):
		pass

	##############################################################
	######## Generic function to create all bar ##################
	##############################################################
	def makeBars(self, df, idx, first_idx):
		''' Create candlestick Bar (for any bar type) given time based dateframe and idx
	    # args
	        df: time based dataframe (candlestick)
	        first_idx: index of the first candlestick
	        idx: list of indices where a candlestick ends
	        mode: specifies the mode of operation
	        
	    # returns
	        candlestick bars: for any data (volumeBar) 
	    '''

		col_names =  ['DateStart', 'DateStop', 'Open', 'High', 'Low', 'Close', 'Volume','Price']
		first_idx = 0
		data_list = []
		for counter,last_idx in enumerate(idx):
		
			frame=df.iloc[first_idx:last_idx]

			DateStart=frame.Date.iloc[0]
			DateStop=frame.Date.iloc[-1]
			o=frame.Open.iloc[0]
			h=frame.High.max()
			l=frame.Low.min()
			c=frame.Close.iloc[-1]
			vol=frame.Volume.sum()
			price=frame.Price.mean()
		
			data_list.append([DateStart, DateStop, o, h, l, c, vol,price])
			first_idx=last_idx

		df=pd.DataFrame(data_list,columns=col_names)
		df['DateStart']=pd.to_datetime(df['DateStart'])
		df['DateStop']=pd.to_datetime(df['DateStop'])


		return(df.set_index('DateStart'))

	##############################################################
	######## Specific function for creating Volume bar ###########
	##############################################################

	def CreateVolumeBar(self,df, VolColName, VolThrhld, first_idx):		
		''' This function creates volumebar
		# args
			df: time based dataframe (candlestick)
			first_idx: index of the first candlestick
			VolColName: name for volume data
			VolThrhld: threshold value for volume
		# returns
			Volume based candlestick bars
		'''

		idx = self.volumeBarIdx(df, VolColName, VolThrhld)
		return self.makeBars(df, idx, first_idx)

	def volumeBarIdx(self, df, VolColName, VolThrhld):
		''' compute volume bars    
		# args
			df: pd.DataFrame()
			VolColName: name for volume data
			VolThrhld: threshold value for volume
		# returns
			idx: list of indices where a volume based candlestick ends
		'''
		t = df[VolColName]
		ts = 0
		idx = []
		for i, x in enumerate(t):
			ts += x
			if ts >= VolThrhld:
			    idx.append(i)
			    ts = 0
			    continue
		return idx    

	##############################################################
	######## Specific function for creating Dollar bars ##########
	##############################################################

	def CreateDollarBar(self,df, PriceColName, PriceThrhld, first_idx):		
		''' This function creates dollarbar
		# args
			df: time based dataframe (candlestick)
			first_idx: index of the first candlestick
			PriceColName: name for price data
			PriceThrhld: threshold value for price
		# returns
			Dollar based candlestick bars
		'''

		idx = self.dollarBarIdx(df, PriceColName, PriceThrhld)
		return self.makeBars(df, idx, first_idx)

	def dollarBarIdx(self, df, PriceColName, PriceThrhld):
		''' compute dollar bars    
		# args
			df: pd.DataFrame()
			PriceColName: name for price data
			PriceThrhld: threshold value for price
		# returns
			idx: list of indices where a dollar based candlestick ends
		'''
		price = df[PriceColName]*df['Volume']
		ts = 0
		idx = []
		for i, x in enumerate(price):
			ts += x
			if ts >= PriceThrhld:
			    idx.append(i)
			    ts = 0

			    continue
		return idx    

	##############################################################
	######## Specific function for creating Tick bars ###########
	##############################################################

	def CreateTickBar(self,df, PriceColName, TickThrhld, first_idx):		
		''' This function creates ticksbar
		# args
			df: time based dataframe (candlestick)
			first_idx: index of the first candlestick
			PriceColName: name for price data
			TickThrhld: threshold value for No. of Transactions
		# returns
			Dollar based candlestick bars
		'''

		idx = self.tickBarIdx(df, PriceColName, TickThrhld)
		return self.makeBars(df, idx, first_idx)

	def tickBarIdx(self, df, PriceColName, TickThrhld):
		''' compute tick bars    
		# args
			df: pd.DataFrame()
			PriceColName: name for price data
			TickThrhld: threshold value for No. of Transactions
		# returns
			idx: list of indices where a tick based candlestick ends
		'''
		price = df[PriceColName]*df['Volume']
		#ts = 0
		idx = []
		for i, x in enumerate(price):
			#ts += 1
			if i%TickThrhld == 0 and i!=0:
			    idx.append(i)
			    #ts = 0
			    continue
		return idx

	##############################################################
	######## Specific function for creating Time bars ###########
	##############################################################

	def CreateTimeBar(self,df, TimeColName, TimeThrhld, first_idx):		
		''' This function creates timesbar
		# args
			df: time based dataframe (candlestick)
			first_idx: index of the first candlestick
			TimeColName: name for time data
			TimeThrhld: threshold value for Time in seconds
		# returns
			Time based candlestick bars
		'''

		idx = self.timeBarIdx(df, TimeColName, TimeThrhld)
		return self.makeBars(df, idx, first_idx)

	def timeBarIdx(self, df, TimeColName, TimeThrhld):
		''' compute time bars    
		# args
			df: pd.DataFrame()
			TimeColName: name for time data
			TimeThrhld: threshold value for Time
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
		return idx


	def createBars(self,df,name_of_bar,thrhld,first_idx=0):
    
		if name_of_bar == 'volume':
		    df = self.CreateVolumeBar(df,'Volume',thrhld,first_idx)
		if name_of_bar == 'dollar':
		    df = self.CreateDollarBar(df,'Price',thrhld,first_idx)
		if name_of_bar == 'tick':
		    df = self.CreateTickBar(df,'Price',thrhld,first_idx)
		if name_of_bar == 'time':
		    df = self.CreateTimeBar(df,'Date',thrhld,first_idx)

		return df












