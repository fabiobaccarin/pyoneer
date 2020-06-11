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


class SpearmanCorrSelector:
    
    def __init__(self, corr_cutoff: float=.75, pval_cutoff: float=1.0):
        self.corr_cutoff = corr_cutoff
        self.pval_cutoff = pval_cutoff
        
    def _corr_filter(self, X: pd.DataFrame, corr: pd.DataFrame,
                     cutoff: float, ranking: list) -> pd.DataFrame:
        keep = set(ranking)
        for i in ranking:
            keep -= set([j for j in ranking
                         if ranking.index(i) < ranking.index(j)
                         and corr.at[i, j] > cutoff])
        return X[keep]
    
    def filter(self, df: pd.DataFrame, y_col: str) -> pd.DataFrame:
        X = df.drop(columns=y_col)
        y = df[y_col]
        rk_ = self.rank(X, y)
        pval_filter = rk_['p_value'] < self.pval_cutoff
        too_good_filter = rk_['assoc'].abs() < self.too_good_to_be_true
        rk_ = rk_[pval_filter & too_good_filter]
        rk = rk_['rank'].sort_values().index.to_list()
        corr = X.corr(method='spearman').abs()
        for r in np.sort(np.linspace(0, self.assoc_cutoff))[::-1]:
            X_new = self._corr_filter(X, corr, r, rk)
            v = vif(X_new).max()
            if v < self.vif_cutoff:
                return X_new


class PValue:
    """ Calculates p-values for variables to use for feature selection and
        ranking. It uses the following conventions:
            1. If the response is categorical and the predictor is also
               categorical, the p-value refers to the result of a Fisher's
               exact test
            2. If the response is categorical and the predictor is continuous,
               the p-value refers to the result of a Welch test where the
               null hypothesis is that the mean value of the predictor is
               the same for both classes of the response
            3. If the response is continuous and the predictor is categorical,
               the p-value refers to the result of a Welch test where the
               null hypothesis is that the mean value of the response is the
               same in both classes of the predictor
    
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
        
        pvalue: pandas.Series
            P-values for variables in X. The variable names are the indexes
            of the series
            
        pvalue_ci: pandas.DataFrame
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
        
    def _setup(self, X: pd.DataFrame, y: pd.Series, y_categorical: bool,
               catvars: list) -> None:
        guards.not_dataframe(X, 'X')
        guards.not_series(y, 'y')
        guards.not_iterable(catvars, 'catvars')
        if catvars == [] and not y_categorical:
            raise ValueError('Cannot compute for continuous variables only. Either y must be categorical or some column in X must be listed in `catvars`')
        
        self._cols = X.columns.to_list()
        
        self.y_categorical = y_categorical
        self.catvars = catvars
    
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
        
        self._setup(X, y, y_categorical, catvars)
        
        self.pvalue = X.apply(self._pvalue, args=(y,))
        self.pvalue.name = 'pvalue'
        
        self.diff_means = (X[[col for col in self._cols
                             if self._cont_cat_pair(col)]]
                           .apply(self._diff_means, args=(y,)))
        self.diff_means.name = 'diff_means'
        
        if self.catvars != []:
            self.odds_ratio = (X[[col for col in self.catvars
                                 if self.y_categorical]]
                               .apply(self._odds_ratio, args=(y,)))
            self.odds_ratio.name = 'odds_ratio'


    def fit_ci(self, X: pd.DataFrame, y: pd.Series, y_categorical: bool,
               catvars: list=[], weights: pd.Series=None) -> None:
        """ Fits to the attribute matrix X and response y, calulating
            confidence infervals using bootstrapping
        
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
                
            weights: pandas.Series, default None
                If specified, it is a vector of weights to sample. Observations
                with higher values in this vector are more likely to be
                sampled
        """
        
        self._setup(X, y, y_categorical, catvars)
        guards.is_none(self.size, 'size')
        guards.is_none(self.samples, 'samples')
        
        vals = []
        for s in range(self.samples):
            X_s = X.sample(self.size, weights=weights, replace=True)
            y_s = y[X.index]
            vals.append(X_s[self._cols].apply(self._pvalue, args=(y_s,)).T)
        self.pvalue_ci = pd.DataFrame(vals)
        
    @staticmethod
    def scale(vals: pd.Series) -> pd.Series:
        """ Returns the negative of the p-values in a log 10 scale
        
            Parameters
            ----------
            vals: pandas.Series
                vector of p-values to transform
                
            Returns
            -------
            pandas.Series:
                Transformed p-values
        """
        return -np.log10(vals)