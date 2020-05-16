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


def pd_curve(predictor: abc.Callable, data: pd.DataFrame, feature: str,
             min_val: float, max_val: float, n: int=None, step: float=None,
             make_val: abc.Callable=None, make_val_args: tuple=(),
             estimator: abc.Callable=np.mean, pred_col: int=0,
             y_hat_label: str='y_hat') -> pd.DataFrame:
    ''' Calculates a partial dependence (PD) curve for a given model and a
        given feature.
        
        Parameters
        ----------
        predictor: function
            Any machine learning model method used to make predictions
            
        data: pandas.DataFrame
            Dataframe containing ONLY the features' values in the data that
            one wants to use to make the PD curve
            
        feature: str
            Name of the column in `data` corresponding to the feature of
            interest
            
        predict_method: str
            Name of the predict method used by `model` to generate predictions
            
        min_val: float
            Minimum value of the grid of values that one wants to use to derive
            the curve
            
        max_val: float
            Maximum value of the grid of values that one wants to use to derive
            the curve
            
        n: int, default None
            Number of elements in the grid of values. It is used to generate
            the grid, which is an array of `n` equally spaced numbers between
            `min_val` and `max_val` (inclusive)
            
        step: float, default None
            Step between elements in the grid of values. It is used to generate
            the grid, which is an array of an `step`-spaced interval between
            `min_val` and `max_val` (exclusive)
            
        make_val: function, default None
            Function to be used to set the values to be tested. Use this
            argument to specify functions with you need to make some
            transformation to your data, like centering and scaling, or
            applying another function. The function must operate on the values,
            not on the whole array
            
        make_val_args: tuple, default ()
            Tuple of positional arguments to be passed to `make_val` after
            the first argument
            
        estimator: function, default numpy.mean
            Function that operates on a numpy `ndarray` or a numpy `array`
            and returns a single value. Used to aggregate the values for
            `data` associated with a particular value of `feature`
            
        pred_col: int, default 0
            Index of the prediction column to use. Set this value for cases
            where the predict method of `model` returns more than one column,
            like `predict_proba` in `sklearn` classifiers
            
        Returns
        -------
        pdc: pandas.DataFrame
            Dataframe with two columns. The first contains the `feature` values 
            used to generate the predictions. The second column contains the 
            corresponding predictions
    '''
    
    guards.not_callable(predictor, 'predictor')
    guards.not_dataframe(data, 'data')
    guards.not_callable(estimator, 'estimator')
    guards.not_both_none(n, step, ['n', 'step'])
    
    pdc = {}
    
    if n is not None:
        grid = np.linspace(min_val, max_val, n).tolist()
    else:
        grid = np.arange(min_val, max_val, step).tolist()
    
    for val in grid:
        df = utils.set_column_val(data, feature, val)
        if make_val:
            guards.not_callable(make_val, 'make_val')
            df[feature] = df[feature].apply(make_val, args=make_val_args)
        pdc[val] = estimator(predictor(df)[:, pred_col])
    
    return (pd.Series(pdc, name=y_hat_label)
            .reset_index().rename(columns={'index': feature}))
    

def _delta_metric(after, before, deviation):
    if deviation == 'arithmetic':
        res = after - before
    elif deviation == 'multiplicative':
        res = after / before
        
    return res


def permutation_importance(X: pd.DataFrame, y: pd.Series, eval_metric: abc.Callable,
                           before_metric_val: float, permutations: int=1000,
                           pred_col: int=0, estimator: abc.Callable=np.mean,
                           deviation: str='arithmetic', use_features: list=None,
                           long: bool=False) -> pd.DataFrame:
    ''' Calculates the permutation importance of features in a dataframe,
        according to a specified metric.
        
        Parameters
        ----------
        X: pandas.DataFrame
            Matrix of attributes of the model
            
        y: pandas.Series
            Vector of observed values of the target (aka dependent variable)
            
        eval_metric: function X, y -> float
            Function corresponding to the metric which is used to anchor
            the exercise. Must receive the feature matrix X as its first argument
            and the vector of observed values of the target y as its second
            argument
            
        before_metric_val: float
            Previous value of the evaluation metric. That is, the value of the
            evaluation metric before permutation
            
        permutations: int, default 1000
            Number of permutations to perform for each feature
            
        pred_col: int, default 0
            Index of the prediction column to use. Set this value for cases
            where the predict method of `model` returns more than one column,
            like `predict_proba` in `sklearn` classifiers
            
        estimator: function, default numpy.mean
            Function to aggregate the results between features. Must return
            a single number
            
        deviation: str, default 'arithmetic'
            The type of deviation to use when measuring the impact of
            permutation on the evaluation metric. Can be either 'arithmetic'
            or 'multiplicative'. In the first case, the difference of the
            values before and after permutation is calculated. If 'multiplicative',
            the values are divided. The value after permutation is always the
            first operand in both cases
            
        use_features: list, default None
            List of features' names in X to apply the exercise. If `None`, the
            exercise will be applied to all columns
            
        long: bool, default False
            Indicates that the output should be returned in 'long' format. If
            `True`, the result is a `pandas.DataFrame` with two columns and
            n x `permutations` rows, where n is the number of columns in `X`.
            The first column, named 'var' contains the name of the variable;
            the second column, named 'pfi' contains the corresponding feature
            importance value for that variable in a given permutation
            
        Returns
        -------
        pfi: pandas.DataFrame
            Dataframe with the number of rows corresponding to the number
            of `permutations` and the same number of columns of `X`. The
            values are the PFIs corresponding to the given permutation for
            the respective variable indicated by the column label. If `long`,
            this formatted is changed to a 'long' format (see `long`)
    '''
    
    guards.not_dataframe(X, 'X')
    guards.not_series(y, 'y')
    guards.not_callable(eval_metric, 'eval_metric')
    guards.not_callable(estimator, 'estimator')
    guards.not_in_supported_values(deviation, ['arithmetic', 'multiplicative'])
        
    pfi = {}
    
    if use_features is not None:
        guards.not_iterable(use_features, 'use_features')    
        features = use_features
    else:
        features = X.columns.to_list()
    
    for feature in features:
        values = []
        for _ in range(permutations):
            X_permut = utils.shuffle_dataframe_column(X, feature)
            metric = eval_metric(X_permut, y)
            values.append(_delta_metric(metric, before_metric_val, deviation))
        pfi[feature] = values
        
    pfi = pd.DataFrame(pfi)
    
    return (pfi if not long
            else utils.DataFrameFormatter.matrix_to_long(pfi, 'pfi'))
            