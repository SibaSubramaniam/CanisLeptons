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


import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import IFrame
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True)



class Visualizer():

	def share_plot(self,*args):
		lst=[]
		for i in range(len(args)):
			lst.append('ax'+str(i))

		tup=tuple(lst)
		fig, tup = plt.subplots(nrows=len(tup), sharex=True)

		for ax,val in zip(tup,args):
			ax.plot(val)
			ax.set_title(val.name)

	def marker_plot(self,labels):

		close=labels['close']
		ret=labels['return']
		vol=labels['volatility']
		labs=labels['label']
		up=close[labs==1.0]
		hold=close[labs==0.0]
		down=close[labs==-1.0]

		f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='all')
		f.suptitle('Analysis of Close, Returns,Volatility and Labels')

		close.plot(ax=ax1, alpha=.5)
		up.plot(ax=ax1,ls='',marker='^', markersize=7,
		alpha=0.75, label='buy', color='g')
		hold.plot(ax=ax1,ls='',marker='o', markersize=7,
		alpha=0.75, label='hold', color='b')
		down.plot(ax=ax1,ls='',marker='v', markersize=7, 
		alpha=0.75, label='sell', color='r')
		ax1.legend()
		ax1.set_title('Labels')

		ax2.plot(close)
		ax2.set_title('Close')

		ax3.plot(ret)
		ax3.set_title('Returns')

		ax4.plot(vol)
		ax4.set_title('Volatility')


		return (ax1,ax2,ax3,ax4)


	def compare_labels(self,labels,un_labels,diff=False):

		close=labels['close']
		labs=labels['label']
		up=close[labs==1.0]
		hold=close[labs==0.0]
		down=close[labs==-1.0]

		un_close=un_labels['close']
		un_labs=un_labels['label']
		un_up=un_close[un_labs==1.0]
		un_hold=un_close[un_labs==0.0]
		un_down=un_close[un_labs==-1.0]

		if diff:
			up_d=un_close[un_labs==2.0]
			down_d=un_close[un_labs==-2.0]
			hold_d=un_close[un_labs==10.0]




		fig,(ax2,ax4)=plt.subplots(nrows=2,sharex=True,sharey=diff)
		ax2.set_title('Labels Triple Barrier')
		close.plot(ax=ax2, alpha=.5)
		
		if not up.empty:
			up.plot(ax=ax2,ls='',marker='^', markersize=7,
				 alpha=0.75, label='buy', color='g')
		if not hold.empty:
			hold.plot(ax=ax2,ls='',marker='o', markersize=7,
				alpha=0.75, label='hold', color='b')
		if not down.empty:
			down.plot(ax=ax2,ls='',marker='v', markersize=7, 
				   alpha=0.75, label='sell', color='r')
		
		ax2.legend()
		ax4.set_title('Labels predicted')
		un_close.plot(ax=ax4, alpha=.5)
		
		if not un_up.empty:
			un_up.plot(ax=ax4,ls='',marker='^', markersize=7,
				 alpha=0.75, label='buy', color='g')
		if not un_hold.empty:
			un_hold.plot(ax=ax4,ls='',marker='o', markersize=7,
				alpha=0.75, label='hold', color='b')
		if not un_down.empty:
			un_down.plot(ax=ax4,ls='',marker='v', markersize=7, 
				   alpha=0.75, label='sell', color='r')
		if diff:
			if not up_d.empty:
				up_d.plot(ax=ax4,ls='',marker='^', markersize=7,
				 alpha=0.75, label='def buy', color='c')
			if not down_d.empty:
				down_d.plot(ax=ax4,ls='',marker='v', markersize=7,
				alpha=0.75, label='def sell', color='m')
			if not hold_d.empty:
				hold_d.plot(ax=ax4,ls='',marker='o', markersize=7,
				alpha=0.75, label='def sell', color='k')

		ax4.legend()


	def df_plot(self,df):
		df.plot()


	###############################################################
	######## Compare Sharpe ratio labels and close price ##########
	###############################################################

	def vis_sharpe_ratio(self,df):
		"""
		args:
			df: dataframe with atleast Close,Return,sharpe_ratio,label columns
		returns:
			compare plot of all 3
		"""
		
		cl = go.Scatter(
			x=df.index,
			y=df['Close'],
			mode='lines+markers',
			name = 'Close'
		)

		ret = go.Scatter(
			x=df.index,
			y=df['Return'],
			mode='lines+markers',
			name = 'Return'
		)
		
		sr = go.Scatter(
			x=df.index,
			y=df['sharpe_ratio'],
			mode='lines+markers',
			name = 'SR'
		)
		
		# lb1 = df[df['label']>0]
		# lb2 = df[df['label']<0]
		
		l1 = go.Scatter(
			x=df.index,
			y=df['label'],
			mode='markers',
			name = 'Labels',
			
			marker = dict(
			color = 'rgb(206, 61, 0)')
			
		)
		
		
		# data = [cl,l1,l2]
		
		# iplot(data)
		
		

		fig = tools.make_subplots(rows=4, shared_xaxes=True)

		fig.append_trace(cl, 1, 1)
		fig.append_trace(ret, 2, 1)
		fig.append_trace(sr, 3, 1)
		fig.append_trace(l1, 4, 1)
		# fig.append_trace(l2, 5, 1)
		
		fig['layout'].update(height=600, width=800,title='Sharpe ratio check')
		return iplot(fig)

	


	###############################################################
	######## Visualise Close and regions ##########################
	###############################################################

	def view_close(self,df):
		trace = go.Scatter(
			x = df.index,
			y = df['close'],
			mode = 'lines',
			name='Close'
		)
		
		data = [trace]
		return iplot(data)




	###############################################################
	######## Visualise labels in certain Area  ####################
	###############################################################


	def trace_return(self,X,y,mode='markers',name=None,color=None):
		"""
		Helper to return plotly trace 
		"""
		
		t = go.Scatter(
			x = X,
			y = y,
			mode = mode,
			name = name,
			marker = dict(
				color = color
		))
		
		return t

	def labels_check(self,df,area=None):
		"""
		Plots Labels (+1, -1, 0) on close price
		
		args:
			df: Dataframe consisting Labels+Features 
			columns required (Close,label)
		returns:
			plotly plot of labels
		"""
		
		### Only certain area plot according to index of dataframe ###
		if area is not None:
			temp_df = df.iloc[area[0]:area[1],:]
			df = temp_df.copy(deep=True)
			
			
		neg_df = df[df['label']==-1]
		pos_df = df[df['label']==1]
		zer_df = df[df['label']==0]
		# print(neg_df.shape,pos_df.shape,zer_df.shape)
		data = []
		
		list_df = [neg_df,pos_df,zer_df]
		names = ['-1', '+1', '0']
		colors = ['rgb(255,0,0)', 'rgb(0,255,0)', 'rgb(0,0,255)']
		
		
		### Close plots ###
		cl = t = go.Scatter(
			x = df.index,
			y = df['close'],
			mode = 'lines',
			name = "Close",
			marker = dict(
				color = 'rgb(244, 203, 66)'
		))
		
		data.append(cl)
		
		
		### Label plots ###
		for i,j,k in zip( list_df, names ,colors, ):
			X = i.index
			# Ask siba to change 'close' to 'Close'
			y = i['close']
			
			# self,X,y,mode='markers',name=None,color=None
			trace = self.trace_return(X,y,'markers',j,k)
			data.append(trace)
			
		fig = go.Figure(data=data)
		fig['layout'].update(height=600, width=1200, title='Labels -1, +1, 0')
		# print(data)
		return plot(fig,filename='Label Check.html')



	###############################################################
	######## Visualise All Features+Labels in certain Area  #######
	###############################################################


	def trace_return_features(self,X,y,name=None):
		"""
		Helper to return plotly traces of features
		"""
		t = go.Scatter(
			x = X,
			y = y,
			mode = 'lines+markers',
			name = name,
		)

		return t

	def features_check(self,df,area=None,plot_name=None):
		"""
		Plots All features in dataframe
		(it also has capability to print labels and pred_labels)

		args:
			df: Dataframe consisting Labels+Features 
			columns required (Close,label)
		returns:
			plotly plot of all columns in dataframe
		"""

		### Only certain area plot according to index of dataframe ###
		if area is not None:
			temp_df = df.iloc[area[0]:area[1],:]
			df = temp_df.copy(deep=True)

		
		# print(neg_df.shape,pos_df.shape,zer_df.shape)
		data = []
		n = df.columns.shape[0]
		names = df.columns
		colors = []
		for i in range(df.columns.shape[0]):
			c = tuple(np.random.choice(range(256), size=3))
			c = 'rgb'+str(c)
			colors.append(c)


		
		fig = tools.make_subplots(rows=n, cols=1,
							  shared_xaxes=True)
		
		nm_list = list(names)
		### Label plots ###
		j=1
		for i in nm_list:
			
			X = df[i].index
			# Ask siba to change 'close' to 'Close'
			y = df[i]

			# self,X,y,mode='markers',name=None,color=None
			trace = self.trace_return_features(X,y,i)
			
			fig.append_trace(trace,j,1)
			j+=1

		# print(data)

		fig['layout'].update(height=1800, width=1200, title='Stacked Subplots of all Features')
		if plot_name is None:
			plot_name = 'Full Check Dataframe.html'
		else:
			plot_name = plot_name+'.html'
		return plot(fig,filename=plot_name)




	###############################################################
	######## Visualise Certain columns in certain Area  ###########
	###############################################################


	def view_certain_cols(self,df,area=None,list_cols=['label'],name='Certain Columns Plot'):
		"""
		Subplots Shared X one below other
		"""
		if area is not None:
			temp_df = df.iloc[area[0]:area[1],:]
			df = temp_df.copy(deep=True)

		
		n=len(list_cols)
		fig = tools.make_subplots(rows=n, cols=1,shared_xaxes=True)
		
		 
		for p,i in enumerate(list_cols,1):
			X = df.index
			y = df[i]
			
			t = self.trace_return_features(X,y,i)
			
			fig.append_trace(t,p,1)
			
		if name is not None:
				name+='.html'
				
		return plot(fig,filename=name)



	###############################################################
	##### GENERIC PLOT FUNCTION (also includes all subplots) ######
	###############################################################



	def generic_plot(self,df,area,list_cols,rc=[1,1],plot_name=None):
		"""
		args:
			df: dataframe to be fed for plotting
			area: list, [start,stop] index of dataframe (region of interest)
			list_cols: list of cols that needs to be visualised from df
			rc: (default [1,1]) parameter for sub-plots multi-row-column plot
			plot_name: name of plot
		
		returns:
			plotly plot
		"""
		n_rows,n_cols = rc[0],rc[1]
		if n_rows*n_cols < len(list_cols):
			print("Enter proper rows and columns according to no. of features")
			return
		
		
		### Only certain area plot according to index of dataframe ###
		if area is not None:
			temp_df = df.iloc[area[0]:area[1],:]
			df = temp_df.copy(deep=True)

			
		### If only single column plot one below other ###
		"""
		if n_cols == 1 :
			plot = self.view_certain_cols(df=df,area=area,list_cols=list_cols,name=plot_name)
			return plot
		"""

		### Plot according to no. of columns ###
		if n_cols >= 1 and n_rows>=1 :
			
			# create a figure of subplots
			fig = tools.make_subplots(rows=n_rows, cols=n_cols,shared_xaxes=True)
			
			for i in range(1,n_rows+1):
				for j in range(1,n_cols+1):
					if not list_cols:
						break
					
					c = list_cols[0]
					t = self.trace_return_features(df.index,df[c],name=c)
					
					try:
						fig.append_trace(t,i,j)
					except:
						# print("Error in appending figures, please check rc parameter")
						pass
					print(i,j,list_cols[0])
					
					
					list_cols.pop(0)
			if plot_name is None:
				plot_name = 'N-Generic-Plot.html'
			else:
				plot_name = plot_name+'.html'
				
			return plot(fig,filename=plot_name)





	###############################################################
	##### GENERIC overlap PLOTs FUNCTION  #########################
	###############################################################

	def generic_overlap_plots(self,df,list_cols,area=None,plot_name=None):
		"""
		Generic method to plot overlap graphs
		args:
			df: Dataframe
			list_cols: list of columns to be plotted in overlap plots
			area: list, [start,stop] index of dataframe (region of interest)
			plot_name: name of plot
		returns:
			plotly graph
		"""
		### Only certain area plot according to index of dataframe ###
		if area is not None:
			temp_df = df.iloc[area[0]:area[1],:]
			df = temp_df.copy(deep=True)

		
		data = []
		for i in list_cols:
			print(i)
			trace = self.trace_return_features(df.index,df[i],i)
			data.append(trace)
		
		
		fig = go.Figure(data)
		fig['layout'].update(title='Overlap Plot')
		if plot_name is None:
			plot_name = 'Overlap-Plot.html'
		else:
			plot_name = plot_name+'.html'
		return plot(fig,filename=plot_name)