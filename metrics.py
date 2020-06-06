#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:33:59 2020

@author: Fabio Baccarin

Metrics module
"""


import pandas as pd
import numpy as np
from pyoneer import guards, utils
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

    scores = utils.make_score(predictor, X, good_score=False)
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


class Associator:
    ''' Calculates association measures between pairs of variables.
        
        The pair of variables can be of three types: continuous-continuous
        (cont-cont), continuous-categorical (cont-cat) or
        categorical-categorical (cat-cat). The corresponding association
        measures used for each case are:
            cont-cont: Spearman's correlation coefficient
            cont-cat: Point biserial correlation coefficient
            cat-cat: Corrected Cramér's V
    '''
    
    def __init__(self, contvars):
        self.contvars = contvars
        
    def contcont(self, x: str, y: str) -> bool:
        ''' Returns `True` if `x` and `y` are both in `self.contvars` '''
        return x in self.contvars and y in self.contvars
    
    def catcat(self, x: str, y: str) -> bool:
        ''' Returns `True` if `x` and `y` are both not in `self.contvars` '''
        return x not in self.contvars and y not in self.contvars
    
    def contcat(self, x: str, y: str) -> bool:
        ''' Returns `True` if either `x` or `y` are not in `self.contvars` '''
        return not self.contcont(x, y) and not self.catcat(x, y)
    
    def assoc(self, var1: pd.Series, var2: pd.Series, signum: bool=False):
        ''' Returns the association measure corresponding to the pair of
            variables `var1` and `var2`
            
            Parameters
            ----------
            var1: pandas.Series
                Vector of values of the first variable
                
            var2: pandas.Series
                Vector of values of the second variable
                
            signum: bool, default False
                If `True`, the method returns a tuple containing the value
                of the appropriate association measure as its first element,
                and the signal of the measure as its second element. If `False`,
                the function returns only the value os the association measure
                as a float
        '''
        
        guards.not_series(var1, 'var1')
        guards.not_series(var2, 'var2')
        
        if self.contcont(var1.name, var2.name):
            val = stats.spearmanr(var1, var2).correlation
        elif self.catcat(var1.name, var2.name):
            val = cramerV(var1, var2)
        else:
            val = stats.pointbiserialr(var1, var2).correlation
            
        return (val, np.sign(val)) if signum else val
        
    def assoc_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' Calculates a matrix of association measures based on the pair's type
                
            Parameters
            ----------
            X: pandas.DataFrame
                Dataframe consisting of measures whose association one wants to
                evaluate
                
            Returns
            -------
            assoc: pandas.DataFrame
                Square matrix of association measures, following the rules for
                pair type above
        '''
        
        guards.not_dataframe(X, 'X')
        
        cols = X.columns.to_list()
        assoc = pd.DataFrame({k: np.nan for k in cols}, index=cols)
        
        for i in cols:
            assoc.at[i, i] = 1
            for j in cols:
                if cols.index(i) < cols.index(j):
                    val = self.assoc(X[i], X[j])
                    assoc.at[i, j] = val
                    assoc.at[j, i] = val
        
        return assoc