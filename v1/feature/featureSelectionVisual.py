'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''


import numpy as np
import pandas as pd

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import IFrame
import plotly.graph_objs as go
import plotly.plotly as py
init_notebook_mode(connected=True)


import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelectionVisual(object):
	"""
	docstring for FeatureSelection Verification and visualisation
	contains methods for visualising 
	"""

	def __init__(self):
		pass




	################################################################################
	##### Specific function to Visualise correlation matrix heatmap ################
	################################################################################

	def corr_heatmap(self,df):
		"""
		Get correlation heatmap
		args:
			df : correlation dataframe
		returns:
			iplot : heatmap of 
		"""
		import plotly.plotly as py
		import plotly.graph_objs as go
		import plotly.tools as tls
		from plotly.offline import iplot, init_notebook_mode
		import warnings
		warnings.filterwarnings(action='ignore')

		init_notebook_mode = True
		# print(df)
		trace = go.Heatmap(z=df.values,
					   x=df.columns,
					   y=df.columns)
		layout = go.Layout(
			width=650,
			height=650
		)

		data=[trace]
		fig = go.Figure(data=data, layout=layout)

		return iplot(fig, filename='labelled-heatmap')




	################################################################################
	##### Specific function to Visualise distribution plots Univariate #############
	################################################################################


	def dist_plot(self,df,col_name='Close'):
		"""
		Method to get distribution plot
		args:
			df : dataframe of features
			col_name : column names in dataframe
			accepted inputs for col_name
			['sma', 'ema', 'dema', 'tema', 'rsi', 'bb_up', 'bb_dn', 'Close', 'sma5',
			'sma10', 'sma15', 'sma20', 'sma25', 'ema5', 'ema10', 'ema15', 'ema20',
			'ema25', 'VPT', 'EMV', 'WilliamsR', 'ROC'],
		returns:
			distribution plot
		"""
		plt.title(col_name.upper()+' distribution plot')
		return sns.distplot(df[col_name].dropna())