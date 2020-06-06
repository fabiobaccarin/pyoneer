#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 08:36:11 2020

@author: Fabio Baccarin

Explainable Artificial Inteligence (XAI) module. Contains tools that help
humans understand complex models
"""

import pandas as pd
import numpy as np
from collections import abc
from pyoneer import guards, utils
            

def deletion_diagnostics(data: pd.DataFrame, y_col: str, base_metric: float,
                         learner, fit: abc.Callable, predict: abc.Callable,
                         aggfunc: abc.Callable=np.mean,
                         deviation: str='arithmetic') -> pd.DataFrame:
    ''' Performs deletion diagnostics for the specified model. It computes
        the difference in predicted values, aggregated according to a specified
        aggregation function
        
        Parameters
        ----------
        data: pandas.DataFrame
            Information to be used for learning
            
        y_col: str
            Name of the column in `data` to containing the target
            supervising the model
            
        base_metric: float
            Value of the diagnostics measure before deletion exercises. Used
            for measuring changes
            
        learner: Any
            Any object corresponding to a learner. It must be already
            initialized with desired hyperparameters values
            
        fit: learner, X, y -> fitted
            Fits the learner to the data. It must have the
            following signature: learner X, y -> fitted. It must return
            the fitted learner
        
        predict: fitted, X -> pandas.Series
            Uses the learner for making predictions. It must
            have the following signature: fitted, X -> pandas.Series. The
            first argument must be the learner, supossed fitted and ready
            to predict
            
        aggfunc: callable, default np.mean
            Function to be used to aggregate `metric` for all data after
            predicting with deletion
        
        deviation: str, default 'arithmetic'
            Type of deviation to use for calculating diagnostics. If
            'arithmetic' (default), the difference is calculated. If
            'multiplicative', the ratio is calculated. It must be one of
            these two
    '''
        
    guards.not_dataframe(data, 'data')
    guards.not_callable(fit, 'fit')
    guards.not_callable(predict, 'predict')
    guards.not_in_supported_values(deviation, ['arithmetic', 'multiplicative'])
    
    influences = {}
    
    for idx in data.index:
        X = data.drop(index=idx, columns=y_col)
        y = data.drop(index=idx)[y_col]
        
        fitted = fit(learner, X, y)
        y_hat = predict(fitted, X)
        
        after = aggfunc(y_hat)
        influences[idx] =  _delta_metric(after, base_metric, deviation)
        
    return pd.Series(influences)