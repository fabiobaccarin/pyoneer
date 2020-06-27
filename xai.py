"""
Explainable Artificial Inteligence (XAI) module. Contains tools that help
humans understand complex models
"""

import typing as t
import pandas as pd
import numpy as np
from pyoneer import guards
from pyoneer import type_aliases as ta
            

def deletion_diagnostics(data: pd.DataFrame, y_col: str, base_metric: float,
        learner: t.Any,
        fit: t.Callable[[t.Any, ta.Matrix, t.Optional[ta.Vector]], t.Any],
        predict: t.Callable[[t.Any, ta.Matrix], ta.Vector],
        aggfunc: t.Callable[[ta.Vector], float],
        deviation: t.Optional[str]='arithmetic') -> pd.DataFrame:
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
            
        fit: Callable[[learner, X, y], learner]
            Fits the learner to the data. It must return the fitted learner
        
        predict: Callable[[learner, X], Union[pandas.Series, numpy.array]]
            Uses the learner for making predictions. The first argument must be 
            the learner, suposed fitted and ready to predict
            
        aggfunc: Callable[[Union[pandas.Series, numpy.array]], float], default 
            np.mean
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
