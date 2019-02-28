import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import pickle

from labelling.labelgenerator import LabelGenerator

class LabelGen():
	def __init__(self):
		self.lbl_ob = LabelGenerator()

	def generate_labels(self,filename,cstk_df,sampling,volatility_threshold,v_bars_duration,barrier_conf,min_return,risk,sign_label):
		
		pkl_labels='saved_runs/Labels_'+filename[30:]+'_'+str(sampling)+'_'+str(volatility_threshold)+'_'+str(v_bars_duration)+'_'+str(barrier_conf)+'_'+str(min_return)+'_'+str(sign_label)
		try:
			full_df = pd.read_pickle(pkl_labels)
		except (OSError, IOError) as e:
			print(e)
			full_df = self.lbl_ob.get_barrier_labels(cstk_df,sampling,volatility_threshold,v_bars_duration,
			                                    barrier_conf,min_return,risk,sign_label)
			full_df.to_pickle(pkl_labels)


		return full_df
