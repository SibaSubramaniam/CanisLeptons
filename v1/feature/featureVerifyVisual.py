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


class VerifyFeature(object):
	""" for VerifyFeature"""

	def __init__(self):
		pass


	def plot_and_verify(self,dt_lbf, vec_lbf, dt_f,vec_f, dt_clp ,vec_clp ):
		# BEFORE INTEGRATING MAKE SURE ALL THE DATE COLUMNS ARE IN PANDAS TO DATETIME OBJECTS
		"""
		args:
			dt_lbf: date Series of features matched with labels
			dt_f: date Series of features NOT matched with labels
			dt_clp: date Series of Close price of dataframe containing all csvs data
			vec_lbf: certain feature Series which is mapped with labels
			vec_f: certain feature Series which is NOT mapped with labels
			vec_clp: close price series of dataframe containing all csvs data 
		returns:
			plot of matched and unmatched labels
			if plot overlaps then verified
		"""
		
		dt_lbf = pd.to_datetime(dt_lbf)
		dt_f = pd.to_datetime(dt_f)
		
		d1 = dt_lbf[0]
		d2 = dt_lbf.iloc[-1]
		
		# print(d1,type(d1))
		# print(d2,type(d2))
		
		# get index ranges
		
		r1 = dt_f.loc[dt_f==d1].index
		r2 = dt_f.loc[dt_f==d2].index

		r1_ = dt_clp.loc[dt_clp==d1].index
		r2_ = dt_clp.loc[dt_clp==d2].index
		
		
		
		# get indices of respective dataframes
		r1,r2 = r1,r2
		r1_ , r2_ = r1_, r2_ 
		
		# print(r1_)
		# print(r2_)
		
		
		
		labelled_features = go.Scatter(
			x=dt_lbf,
			y=vec_lbf,
			mode='markers',
			name = 'downsampled features',
			marker=dict(
			size=5,
			color = 'rgb(0, 102, 255)'
			)
		)
		
		
		unlabelled_features = go.Scatter(
			x=dt_f.iloc[r1:r2+1],
			y=vec_f.iloc[r1:r2+1],
			mode='lines',
			name = 'extracted features',
			line =dict(color = ('rgb(0, 255, 0)'),width = 1)
		)
		
		close = go.Scatter(
			x=dt_clp.iloc[r1:r2+1],
			y=vec_clp.iloc[r1:r2+1],
			mode='lines',
			name = 'Close',
			line =dict(color = ('rgb(255, 80, 80)'),width = 1)
		)
		
		
		
		data = [unlabelled_features,close,labelled_features,]
		
		return iplot(data)




	def verify_bars_indicators(self,df,bar_df,indicator_bar_df,col_name='Close',title='Plot'):
		"""
		Verify various bars and effect of moving average on them
			args:
				df: DataFrame whose bars have been calculated
				bar_df: DataFrame of bars (eg: timebar, dollarbar, etc)
				indicator_bar_df: DataFrame of indicator calculated on bar_df (eg: Moving average of Time bar)
			returns:
				plot of close prices for both the dataframes
		"""
		
		
		close = df[col_name]
		b_close = bar_df[col_name]
		bi_close = indicator_bar_df[col_name]
		

		c = go.Scatter(
				x=df['Date'],
				y=close,
				mode='lines+markers',
				name = 'Close',
				marker=dict(
				size=5,
				color = 'rgb(255,0,0)'
				)
			)
		
		
		b = go.Scatter(
				x=bar_df[bar_df.columns[0]],
				y=b_close,
				mode='lines',
				name = 'Bar',
				marker=dict(
				size=5,
				color = 'rgb(0, 0, 255)'
				)
			)


		bi = go.Scatter(
			x=indicator_bar_df['Date'],
			y=bi_close,
			mode='lines',
			name = 'Features',
			line =dict(color = ('rgb(0, 255, 0)'),width = 1)
		)

		layout = go.Layout(title=title)


		data = [c,b,bi]

		fig = go.Figure(data=data,layout=layout)
		return iplot(fig)