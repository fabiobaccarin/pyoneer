#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:33:59 2020

@author: Fabio Baccarin

Metrics module
"""


import pandas as pd
import numpy as np
from pyoneer import guards
from collections import abc
from sklearn.linear_model import LinearRegression
from scipy import stats as ss


def ks(predictor: abc.Callable, X, y) -> float:
    ''' Returns the Kolmogorov-Smirnov (KS) statistic
    
        Parameters
        ----------
        predictor: abc.Callable
            Any object that can be used as a function to use data to predict an
            outcome
            
        X: numpy.ndarray, pandas.DataFrame
            Attribute matrix used by `predictor` to make predictions about an
            independent variable
            
        y: numpy.array, pandas.Series
            Vector of target values, aka independent variable
            
        Returns
        -------
        float: value of Kolmogorov-Smirnov statistic
    '''
    
    guards.not_callable(predictor, 'predictor')

    scores = pd.crosstab(predictor(X)[:, 1], y)
    for col in [0, 1]:
        scores[col] = scores[col].cumsum() / scores[col].sum()

    return (scores[0] - scores[1]).max() * 100


def mev(X: pd.DataFrame) -> pd.DataFrame:
    ''' Returns the Matrix of Explained Variance (MEV). This is a matrix
        constructed by running a simple linear regression (OLS) with every
        pair of variable in the matrix of attributes X. Then, the R-squared
        statistic is calculated and stored in a square matrix. It is equivalent
        to a correlation matrix for pairs of continuous variables
        
        Parameters
        ----------
        X: pandas.DataFrame
            Matrix of attributes
            
        Returns
        -------
        pandas.DataFrame
            MEV matrix
    '''
    
    guards.not_dataframe(X, 'X')
    
    cols = X.columns.to_list()
    matrix = pd.DataFrame({k: np.nan for k in cols}, index=cols)
    reg = LinearRegression()
    
    for i in cols:
        matrix.at[i, i] = 1
        for j in cols:
            if cols.index(i) < cols.index(j):
                x = np.asarray(X[j]).reshape(-1, 1)
                r_sq = reg.fit(x, X[i]).score(x, X[i])
                matrix.at[i, j] = r_sq
                matrix.at[j, i] = r_sq
            
    return matrix


def _vif(r2: float) -> float:
    ''' Returns the variance inflator factor (VIF) '''
    
    return 1/(1 - r2) if r2 < 1 else np.inf


def vif(X: pd.DataFrame) -> pd.Series:
    ''' Returns the Variance Inflator Factor (VIF) for all variables in X
        
        Parameters
        ----------
        X: pandas.DataFrame
            Matrix of attributes
            
        Returns
        -------
        pandas.Series
            VIF vector for all columns in X
    '''
    
    guards.not_dataframe(X, 'X')
    
    results = {}
    reg = LinearRegression()
    
    for col in X.columns.to_list():
        others = X.drop(columns=col)
        r_sq = reg.fit(others, X[col]).score(others, X[col])
        results[col] = _vif(r_sq)
            
    return pd.Series(results, name='vif')


def cramerV(x: pd.Series, y: pd.Series) -> float:
    ''' Cramér's V association statistic
    
        Parameters
        ----------
        x, y: numpy.array, pandas.Series
            Variables whose association one wants to measure
            
        Returns
        -------
        float:
            Cramer's V value
    '''
    
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def yule_q(odds_ratio: float) -> float:
    return (odds_ratio - 1) / (odds_ratio + 1)


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