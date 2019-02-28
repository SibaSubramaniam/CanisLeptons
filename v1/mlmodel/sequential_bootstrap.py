'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import numpy as np


class sequentialBootstrap(object):
    
    """
    This class contain functions for reducing observation redundancy i.e. oversampling.
    
    """
    def __init__(self):
        pass
    
    
    def getIndMatrix(self, barIx,t1):
        '''
        compute indicator matrix
        #args
            barIx: the index of bars
            t1: the pandas Series, defined by an index containing the time at which the features are
                observed, and a values array containing the time at which the label is determined.
        #returns
            binary matrix indicating what (price) bars influence the label for each observation.
        
        '''
        indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
        for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
        return indM
    
    
    def getAvgUniqueness(self, indM):
        '''
        compute Average uniqueness from indicator matrix
        #args
            indM: indicator matrix return by getIndMatrix function (binary matrix indicating what 
                  (price) bars influence the label for each observation).
    
        #returns
            average uniqueness of each observed feature
        '''
        c=indM.sum(axis=1) # concurrency
        u=indM.div(c,axis=0) # uniqueness
        avgU=u[u>0].mean() # average uniqueness
        return avgU
    
    def seqBootstrap(self, indM,sLength=None):
        '''
        Generate a sample via sequential bootstrap
        #args:
            indM: indicator matrix return by getIndMatrix function (binary matrix indicating what 
                  (price) bars influence the label for each observation).
        #returns:
            a sequential bootstrap
        '''
        if sLength is None:sLength=indM.shape[1]
        phi=[]
        while len(phi)<sLength:
            avgU=pd.Series()
            for i in indM:
                indM_=indM[phi+[i]] # reduce indM
                avgU.loc[i]=self.getAvgUniqueness(indM_).iloc[-1]
            prob=avgU/avgU.sum() # draw prob
            phi+=[np.random.choice(indM.columns,p=prob)]
        return phi
    
    
    