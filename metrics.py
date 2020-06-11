#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:33:59 2020

@author: Fabio Baccarin

Metrics module
"""


import pandas as pd
import numpy as np
import warnings
from pyoneer import guards
from pyoneer import warnings as w
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


class Associator:
    
    def __init__(self, catvars: list, assoc_cutoff: float=.75,
                 vif_cutoff: float=5.0, pval_cutoff: float=.99,
                 too_good_to_be_true: float=.99):
        self.catvars = catvars
        self.assoc_cutoff = assoc_cutoff
        self.vif_cutoff = vif_cutoff
        self.pval_cutoff = pval_cutoff
        self.too_good_to_be_true = too_good_to_be_true
        
    def _catcat(self, x: str, y: str) -> bool:
        return x in self.catvars and y in self.catvars
    
    def assoc(self, var1: pd.Series, var2: pd.Series) -> (float, float):
        try:
            if self._catcat(var1.name, var2.name):
                odds_ratio, pval = ss.fisher_exact(pd.crosstab(var1, var2))
                r = yule_q(odds_ratio)        
            else:
                r, pval = ss.spearmanr(var1, var2)
            return r, pval
        except ValueError:
            warnings.warn(w.ASSOC_WARNING.format(var1.name, var2.name),
                          RuntimeWarning)
            return np.nan, np.nan
    
    def corr(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns.to_list()
        matrix = pd.DataFrame({k: np.nan for k in cols}, index=cols)
        for i in cols:
            matrix.at[i, i] = 1
            for j in cols:
                if cols.index(i) < cols.index(j):
                    r = self.assoc(df[i], df[j])[0]
                    matrix.at[i, j] = r
                    matrix.at[j, i] = r
        return matrix
    
    def _corr_filter(self, X: pd.DataFrame, corr: pd.DataFrame,
                     cutoff: float, ranking: list) -> pd.DataFrame:
        keep = set(ranking)
        for i in ranking:
            keep -= set([j for j in ranking
                         if ranking.index(i) < ranking.index(j)
                         and corr.at[i, j] > cutoff])
        return X[keep]
    
    def rank(self, X: pd.DataFrame, y: pd.Series,
             apply_cutoffs: bool=True) -> pd.DataFrame:
        data = {'assoc': [], 'pvalue': [], '-log10(pvalue)': []}
        cols = X.columns.to_list()
        for var in cols:
            r, pval = self.assoc(X[var], y)
            pval2 = -np.log10(pval)
            data['assoc'].append(r)
            data['pvalue'].append(pval)
            data['-log10(pvalue)'].append(pval2)
        df = pd.DataFrame(data, index=cols)
        if apply_cutoffs:
            pval_filter = df['pvalue'] < self.pval_cutoff
            too_good_filter = df['assoc'].abs() < self.too_good_to_be_true
            df = df[pval_filter & too_good_filter]
        df['rank'] = df['assoc'].abs().rank(ascending=False, method='dense')
        
        return df
    
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