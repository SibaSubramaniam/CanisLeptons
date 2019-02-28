'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import numpy as np
from .multiprocess import mpPandasObj


class LabelGenerator(object):
	def __init__(self):
		pass

	def get_volatility(self,df,span):
    
		tdy=df
		ystdy=df.shift(1)

		#calculating return
		ret=((tdy/ystdy)-1).dropna()
		#ret=ret[1:]


		#calculating volatility
		vol=ret.ewm(span=span).std()

		return ret,vol
	

	def vertical_barrier(self,close,timestamps,span):
    
		#calculating vertical bars
		t1=close.index[span:]
		t1=pd.Series(t1,index=timestamps[:t1.shape[0]])
		return t1


	def triple_barrier_single(self,close, events, sltp, molecule=None):
    
		# Sample a subset with specific indices
		if molecule is not None:
		    _events = events.loc[molecule]
		else:
		    _events = events
		    
		touch_idx = pd.DataFrame(index=_events.index)

		# Set Stop Loss and Take Profit
		if sltp[0] > 0:
		    sls = -sltp[0] * _events["trgt"]
		else:
		    # Switch off stop loss
		    sls = pd.Series(index=_events.index)
		if sltp[1] > 0:
		    tps = sltp[1] * _events["trgt"]
		else:
		    # Switch off profit taking
		    tps = pd.Series(index=_events.index)

		# Replace undefined value with the last time index
		vertical_lines = _events["t1"].fillna(close.index[-1])


		for loc, t1 in vertical_lines.iteritems():
		    df = close[loc:t1]
		    # Change the direction depending on the side
		    df = (df / close[loc] - 1) * _events.at[loc, 'side']
		    touch_idx.at[loc, 'sl'] = df[df < sls[loc]].index.min()
		    touch_idx.at[loc, 'tp'] = df[df > tps[loc]].index.min()
		touch_idx['t1'] = _events['t1'].copy(deep=True)

		return touch_idx


	def triple_barrier(self,close, timestamps, sltp=None, trgt=None, min_ret=0,num_threads=16,t1=None,side=None):
    
		#Fix targets to hit first
		if trgt is None:
		    # Switch off horizontal barriers
		    trgt = pd.Series(1 + min_trgt, index=timestamps)
		    sltp = -1   
		else:
		    trgt = pd.Series(trgt, index=timestamps)
		               
		# Get sampled target values and filter by minimum return
		trgt = trgt.loc[timestamps]
		trgt = trgt[trgt > min_ret]

		if len(trgt) == 0:
		    return pd.DataFrame(columns=['t1', 'trgt', 'side'])

		# Get time boundary t1
		if t1 is None:
		    t1 = pd.Series(pd.NaT, index=timestamps)

		# slpt has to be either of integer, list or tuple
		if isinstance(sltp, list) or isinstance(sltp, tuple):
		    _sltp = sltp[:2]
		else:
		    _sltp = [sltp, sltp]

		# Define the side
		if side is None:
		    # Default is 1
		    _side = pd.Series(1., index=trgt.index)
		else:
		    _side = side.loc[trgt.index]

		#create events dataframe for single triple-barrier method
		events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
		events = events.dropna(subset=['trgt'])
		#use multiprocessing to speedup triple-barrier method
		time_idx = mpPandasObj(func=self.triple_barrier_single,
		                       pdObj=('molecule',events.index),
		                       numThreads=num_threads,
		                       close=close,events=events,sltp=_sltp)

		# Skip when all of barrier are not touched
		time_idx = time_idx.dropna(how='all')
		events['type'] = time_idx.idxmin(axis=1)
		events['time'] = time_idx.min(axis=1)
		del events['t1']
		if side is None:
		    events = events.drop('side', axis=1)

		return events

	def get_labels(self,close, events, min_ret=0, sign_label=True, zero_label=0):
    
		# Prices aligned with events
		events = events.dropna(subset=['time'])

		# All used indices
		time_idx = events.index.union(events['time'].values).drop_duplicates()
		close = close.reindex(time_idx, method='bfill')
		# Create out object
		out = pd.DataFrame(index=events.index)
		out['ret'] = close.loc[events['time'].values].values / close.loc[events.index] - 1.

		# Modify return according to the side
		if 'side' in events:
		    out['ret'] *= events['side']
		    out['side'] = events['side']

		# Assign labels
		out['label'] = np.sign(out['ret'])
		out['label'].loc[events['type'] == 'tp'] = 1.0
		out['label'].loc[events['type'] == 'sl'] = -1.0

		out.loc[(out['ret'] <= min_ret) & (out['ret'] >= -min_ret), 'label'] = zero_label
		if 'side' in events:
		    out.loc[out['ret'] <= min_ret, 'label'] = zero_label
		if sign_label:
		    out['label'].loc[events['type'] == 't1'] = zero_label
		out['t1'] = events['time']
		out['type'] = events['type']
		return out



	'''def drop_labels(self,events, minPct=.05):

		"""Return labels for the given DataFrame

	    Parameters
	    ----------
	    events: pd.DataFrame with columns: 't1', 'trgt', 'type', and 'side'
	    minPct: minimum threshold to drop labels

	    Returns
	    -------
	    pd.DataFrame with droped labels
	    """


		# apply weights, drop labels with insufficient examples
		while True:
		    df0=events['bin'].value_counts(normalize=True)
		    if df0.min()>minPct or df0.shape[0]<3:break
		    print('dropped label: ', df0.argmin(),df0.min())
		    events=events[events['bin']!=df0.argmin()]
		return events'''


	def sharp_ratio(self,returns,risk_free_ret,volatility):

		sharp_ratio=(returns-risk_free_ret)/volatility
		return sharp_ratio

	def profit(self,full_df,price):

		close=full_df['close'].values
		labels=full_df['label'].values

		profits=[]
		stocks=0
		price=1

		for i,j in zip(close,labels):
		    if j==1.0:
		        stocks+=price/i
		        price=0
		    if j==-1.0:
		        price+=stocks*i
		        stocks=0
		    profits.append(price)
		    
		df=pd.DataFrame(profits).set_index(full_df.index)
		df.columns=['profit']

		return df
		


	def get_barrier_labels(self,cstk_df,sampling=False,volatility_threshold=10,v_bars_duration=1,
						barrier_conf=[1,1],min_return=0,risk=0,sign_label=False):

		
		#Volatility
		close=cstk_df['Price']
		ret,vol=self.get_volatility(close,volatility_threshold)

		#Downsampling
		if not sampling:
			timestamps=close.index
		
		#Vertical Bars
		v_bars = self.vertical_barrier(close, timestamps,v_bars_duration)

		#Triple-Barrier Method
		events = self.triple_barrier(close, timestamps, sltp=barrier_conf, trgt=vol, min_ret=min_return,num_threads=16,t1=v_bars,side=None)

		#Labels
		labels=self.get_labels(close,events,sign_label=sign_label)
		labels=labels.drop(columns=['t1','type'])

		#sharp_ratio
		sharp_ratio=self.sharp_ratio(ret,risk,vol)
		
		result = pd.concat([close,ret,events,labels,sharp_ratio], axis=1, sort=False)
		result.columns=['close','return','volatility','type','vbars','ret','label','sharp_ratio']
		
		profit=self.profit(result,1)

		result = pd.concat([result,profit], axis=1, sort=False).dropna()
		
		return result









