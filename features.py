#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:03:16 2020

@author: fabiobaccarin

Module for feature processing. Includes feature selection and feature
engineering algorithms
"""

import numpy as np
import pandas as pd
from pyoneer import guards
from scipy import stats as ss

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


class PValue:
    """ Calculates p-values for variables to use for feature selection in
        classification problems
    
        Parameters
        ----------
        size: int, default None
            Size of samples to use for bootstrapping
            
        samples: int, default None
            Number of repetitions of bootstrap to perform
            
        Attributes
        ----------
        
        y_type: str, ('continous', 'categorical')
            Type of response variable
            
        catvars: list of str
            List of variables that should be treated as categorical/dummy
            variables
            
        diff_means: pandas.Series
            Difference in means used in tests
            
        odds_ratio: pandas.Series
            Odds ratios used in tests
        
        pvalues: pandas.Series
            P-values for variables in X. The variable names are the indexes
            of the series
            
        pvalues_ci: pandas.DataFrame
            P-values for variables in X with bootstrapped confidence intervals
    """
    
    def __init__(self, size: int=None, samples: int=None):
        self.size = size
        self.samples = samples
    
    def _cat_cat_pair(self, x: str) -> bool:
        return x in self.catvars and self.y_categorical
    
    def _cont_cat_pair(self, x: str) -> bool:
        return x not in self.catvars and self.y_categorical
    
    def _cat_cont_pair(self, x: str) -> bool:
        return x in self.catvars and not self.y_categorical
    
    def _fisher_exact(self, X: pd.Series, y: pd.Series) -> (float, float):
        return ss.fisher_exact(pd.crosstab(X, y))
    
    def _welch(self, X: pd.Series, y: pd.Series) -> (float, float):
        return ss.ttest_ind(X[y == 0], X[y == 1], equal_var=False)
    
    def _pvalue(self, X: pd.Series, y: pd.Series) -> float:
        if self._cat_cat_pair(X.name):
            _, p = self._fisher_exact(X, y)
        elif self._cont_cat_pair(X.name):
            _, p = self._welch(X, y)
        elif self._cat_cont_pair(X.name):
            _, p = self._welch(y, X)
        
        return p
    
    def _odds_ratio(self, X: pd.Series, y: pd.Series) -> float:
        tab = pd.crosstab(X, y)
        return np.prod(np.diag(tab)) / np.prod(np.diag(np.fliplr(tab)))
    
    def _diff_means(self, X: pd.Series, y: pd.Series) -> float:
        return X[y == 1].mean() - X[y == 0].mean()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, y_categorical: bool,
            catvars: list=[]) -> None:
        """ Fits to the attribute matrix X and response y
        
            Parameters
            ----------
            X: pandas.DataFrame
                Matrix of attributes
                
            y: pandas.Series
                Vector of response (aka target, dependent variable) values to
                use for testing hypothesis
                
            y_categorical: bool
                Whether y is a categorical response
                
            catvars: list of str, default []
                List of categorical variable names. They should be already
                encoded as dummy variables
        """
        
        guards.not_dataframe(X, 'X')
        guards.not_series(y, 'y')
        guards.not_iterable(catvars, 'catvars')
        
        cols = X.columns.to_list()
        
        self.y_categorical = y_categorical
        self.catvars = catvars
        
        self.pvalue = pd.Series({col: self._pvalue(X[col], y) for col in cols},
                                name='pvalue')
        
        self.diff_means = pd.Series({col: self._diff_means(X[col], y)
                                     for col in cols
                                     if self._cont_cat_pair(col)}, name='diff_means')
        
        if self.catvars != []:
            self.odds_ratio = pd.Series({col: self._odds_ratio(X[col], y)
                                         for col in self.catvars
                                         if self.y_categorical}, name='odds_ratio')
