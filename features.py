#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:03:16 2020

@author: fabiobaccarin

Module for feature processing. Includes feature selection and feature
engineering algorithms
"""

import pandas as pd
from pyoneer import guards

def skewness(X: pd.DataFrame, size: int, samples: int=1000) -> pd.DataFrame:
    """ Calculates skewness for every variable (column) in X with bootstrapped
        confidence intervals
    
        Parameters
        ----------
        X: pandas.DataFrame, pandas.Series, numpy.ndarray
            Attribute matrix containing features' values
            
        size: int
            Size of samples in the bootstrap
            
        samples: int, default 1000
            Number of repetitions to perform during bootstrapping
            
        Returns
        -------
        skew: pandas.DataFrame
            Dataframe containing the skewness statistic for every bootstrapped
            sample generated, for each feature in X. It has shape (samples, k),
            where k is the number of columns in X
    """
    
    guards.not_dataframe(X, 'X')
    guards.not_int(size, 'size')
    guards.not_int(samples, 'samples')
    
    skew = pd.DataFrame()

    for _ in range(samples):
        skew = pd.concat([skew, X.sample(size).skew()], axis=1)
    
    return skew.T.reset_index(drop=True)