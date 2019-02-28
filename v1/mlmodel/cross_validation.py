'''
__author__ = "Canis Leptons"
__copyright__ = "Copyright (C) 2018 Canis Leptons"
__license__ = "Private@ Canis Leptons"
__version__ = "1.0"
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

class PurgedKFold(KFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''
    
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        '''
        To initialise the instance
        #args
            n_splits: number of splits for purged cross validation
            t1: the pandas Series, defined by an index containing the time at which the features are
                observed, and a values array containing the time at which the label is determined.
            pctEmbargo: float value, to decide the size for embargoing
        
        '''
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
            
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo

        
        
    def split(self,X,y=None,groups=None):
        '''
        split the dataset into train and test sets involves purging of training set
        #args
            X: input features set
            y: labels
            groups: Group labels for the samples used while splitting the dataset into
            train/test set.

        #returns
            train and test indices 
        
        '''
        
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
            
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)

        test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]

        for i,j in test_starts:
            
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            
            if maxT1Idx<X.shape[0]: # right train (with embargo)
                train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))
                
            yield train_indices,test_indices
          
            
            
class MyPipeline(Pipeline):
    
    def fit(self,X,y,sample_weight=None,**fit_params):
        '''
        To fit model on the dataset
        #args
            X: input features set
            y: labels
            sample_weight: the sum of the attributed returns over the event’s lifespan
            fit_params:  dict of string -> object Parameters passed to the ``fit`` method of each step, where
                         each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.
        #returns
            self : Pipeline
            This estimator        
        '''
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight']=sample_weight
            
        return super(MyPipeline,self).fit(X,y,**fit_params)
    

class HyperParameterTuning(object):
    
    def GridSearch(self, feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.],n_jobs=-1,pctEmbargo=0,**fit_params):
        '''
        This function is to do exhaustive search over specified parameter values for an estimator.
        #args
            feat: input features dataset 
            lbl: labels 
            t1: the pandas Series, defined by an index containing the time at which the features are observed, and a 
                values array containing the time at which the label is determined.
            pipe_clf: given estimator
            param_grid: dict or list of dictionaries, Dictionary with parameters names (string) as keys and lists of
                        parameter settings to try as values, or a list of such dictionaries, in which case the grids
                        spanned by each dictionary in the list are explored. This enables searching over any sequence
                        of parameter settings.
            cv: int, cross-validation generator or an iterable, optional. Determines the cross-validation splitting strategy.
            bagging: list of three float values, where bagging[0]: n_estimators, bagging[1]: max_samples, bagging[2]: 
                     max_features
            n_jobs: int or None, optional (default=None), Number of jobs to run in parallel.
            pctEmbargo: float value, to decide the size for embargoing
            fit_params: dict of string -> object Parameters passed to the ``fit`` method of each step, where
                         each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.
            
        #returns
            
            The given estimator with tuned hyper-parameters
        '''
        if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling
            
        else:scoring='neg_log_loss' # symmetric towards all cases
            
        #1) hyperparameter search, on train data
        inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
        gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,
        scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
        gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline
        
        #2) fit validated model on the entirety of the data
        if bagging[1]>0:
            gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),max_samples=float(bagging[1]),max_features=float(bagging[2]),n_jobs=n_jobs)
            gs=gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
            gs=Pipeline([('bag',gs)])
        
        return gs
    
    
    def RandomizedSearch(self, feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.],
                                                            rndSearchIter=0,n_jobs=-1,pctEmbargo=0,**fit_params):
        '''
        Randomized search on hyper parameters.      
            
        #args
            feat: input features dataset 
            lbl: labels 
            t1: the pandas Series, defined by an index containing the time at which the features are observed, and a 
                values array containing the time at which the label is determined.
            pipe_clf: given estimator
            param_grid: dict or list of dictionaries, Dictionary with parameters names (string) as keys and lists of
                        parameter settings to try as values, or a list of such dictionaries, in which case the grids
                        spanned by each dictionary in the list are explored. This enables searching over any sequence
                        of parameter settings.
            cv: int, cross-validation generator or an iterable, optional. Determines the cross-validation splitting strategy.
            bagging: list of three float values, where bagging[0]: n_estimators, bagging[1]: max_samples, bagging[2]: 
                     max_features
            rndSearchIter: Number of parameter settings that are sampled. It trades off runtime vs quality of the solution.
            n_jobs: int or None, optional (default=None), Number of jobs to run in parallel.
            pctEmbargo: float value, to decide the size for embargoing
            fit_params: dict of string -> object Parameters passed to the ``fit`` method of each step, where
                         each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.
            
        #returns
            
            The given estimator with tuned hyper-parameters
        '''
        if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling
            
        else:scoring='neg_log_loss' # symmetric towards all cases
            
        #1) hyperparameter search, on train data
        inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
        
        if rndSearchIter==0:
            gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
            
        else:
            gs=RandomizedSearchCV(estimator=pipe_clf,param_distributions=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,
            iid=False,n_iter=rndSearchIter)
            gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline
            #2) fit validated model on the entirety of the data
            
            if bagging[1]>0:
                gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps),
                n_estimators=int(bagging[0]),max_samples=float(bagging[1]),max_features=float(bagging[2]),n_jobs=n_jobs)
                gs=gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
                gs=Pipeline([('bag',gs)])
                
        return gs


class Purged_validation(object):
     '''
     This class contain functions to validate an estimator performance using purged K-fold method.
    
        
     '''
    
     def cvScore(self,clf,X,y,t1=None,cv=None,cvGen=None,pctEmbargo=None):
        ''' 
        To calculate cv scores.
        #args
            X: input features set
            y: labels
            t1: the pandas Series, defined by an index containing the time at which the features are observed, and a 
                values array containing the time at which the label is determined.
            cv: int, cross-validation generator or an iterable, optional. Determines the cross-validation splitting strategy.
            cvGen: method to generate cv 
            pctEmbargo: float value, to decide the size for embargoing

        #returns
            train and test indices 
        '''
        if cvGen is None:
            cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged

        train_score=[]
        test_score=[]
    
        for train,test in cvGen.split(X=X):
            fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train])
        
            pred=fit.predict(X.iloc[test,:])
            score_=f1_score(y.iloc[test],pred,average='micro')
            pred1=fit.predict(X.iloc[train,:])
            score_1=f1_score(y.iloc[train],pred1,average='micro')
           
        train_score.append(score_1)
        test_score.append(score_)

        return np.array(train_score), np.array(test_score)
    

     def learning_curve(self, clf, X, y, t1=None, cv=5, pctEmbargo=0., sample_size=[1., 0.8, 0.6, 0.4, 0.2]):
        ''' 
        To plot learning curve.
        #args
            X: input features set
            y: labels
            t1: the pandas Series, defined by an index containing the time at which the features are observed, and a 
                values array containing the time at which the label is determined.
            cv: int, cross-validation generator or an iterable, optional. Determines the cross-validation splitting strategy.
            pctEmbargo: float value, to decide the size for embargoing
            sample_size: list of size of dataset used for calculating scores.
            
        #returns
            plot learning curve
        '''
        train_scores = []
        test_scores = []
        n = len(X)
        for i in sample_size:
            size = int(n*i)
            X = X.iloc[:size]
            y = y.iloc[:size]
            t1 = t1.iloc[:size]
            train, test = self.cvScore(clf, X, y, t1=t1, cv=cv, pctEmbargo=pctEmbargo)
            train = train.mean()
            test = test.mean()
            print('Train score : ', train, ' Test score : ', test)
    
            train_scores.append(train)
            test_scores.append(test)
        
        plt.figure()
        plt.title('Learning curve')
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        plt.plot(sample_size, train_scores, 'o-', color="r", label="Training score")
        plt.plot(sample_size, test_scores, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        
        return plt
    