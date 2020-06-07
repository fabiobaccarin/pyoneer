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
from scipy import stats


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

    scores = predictor(X)[:, 1]
    score_table = pd.crosstab(scores, y)
    goods = score_table.iloc[:, 0].values
    bads = score_table.iloc[:, 1].values
    total_goods = sum(goods)
    total_bads = sum(bads)
    table_size = len(score_table)
    goods_distr = [np.cumsum(goods)[i] / total_goods for i in range(table_size)]
    bads_distr = [np.cumsum(bads)[i] / total_bads for i in range(table_size)]
    diff = [abs(goods_distr[i] - bads_distr[i]) for i in range(table_size)]

    return max(diff) * 100


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
    
    return 1/(1 - r2)


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
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
