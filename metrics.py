#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Metrics module
'''


import pandas as pd
import numpy as np
from pyoneer import guards
from collections import abc
from sklearn.linear_model import LinearRegression
from scipy import stats as ss


def ks(classifier: abc.Callable, X, y) -> float:
    ''' Returns the Kolmogorov-Smirnov (KS) statistic
    
        Parameters
        ----------
        classifier: model
            Any object that has either a `decision_function` or a 
            `predict_proba` method
            
        X: numpy.ndarray, pandas.DataFrame
            Attribute matrix used by `predictor` to make predictions about an
            independent variable
            
        y: numpy.array, pandas.Series
            Vector of target values, aka independent variable
            
        Returns
        -------
        float: value of Kolmogorov-Smirnov statistic
    '''
    
    if hasattr(classifier, 'predict_proba'):
        scores = pd.crosstab(classifier.predict_proba(X)[:, 1], y)
    else:
        scores = pd.crosstab(classifier.decision_function(X), y)
    
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


def odds_ratio(self, x: pd.Series, y: pd.Series) -> float:
    ''' Calculates the odds ratio for two categorical variables x and y. If
        x and y are both binary with values 0 and 1, let T be the contingency
        matrix of those variables (which is 2 x 2). Then the odds ratio is
        the ratio of the product of T's main diagonal entries over the
        product of T's secondary diagonal entries
        
        Parameters
        ----------
        x: pandas.Series
            Variable
            
        y: pandas.Series
            Variable
            
        Returns
        -------
        float:
            Odds ratio
    '''
    tab = pd.crosstab(x, y)
    return np.prod(np.diag(tab)) / np.prod(np.diag(np.fliplr(tab)))


def yule_q(odds_ratio: float) -> float:
    ''' Yule's Q statistic for measuring association between categorical
        variables
        
        Parameters
        ----------
        odds_ratio: float
            Value of odds ratio
            
        Returns
        -------
        float:
            Value of Yule's Q
    '''
    
    return (odds_ratio - 1) / (odds_ratio + 1)


class PValue:
    ''' Class grouping utilities for dealing with p-values '''
    
    @staticmethod
    def bonferroni(pvals: pd.Series, alpha: float=.05) -> pd.Series:
        ''' Bonferroni correction for p-values
        
            Parameters
            ----------
            pvals: pandas.Series
                Vector of p-values to transform
                
            alpha: float, default 0.05
                Level of significance for rejecting the null hypothesis
                
            Returns
            -------
            pandas.Series:
                Whether the null hypothesis is rejected for the corresponding
                p-value
        '''
        
        guards.not_series(pvals, 'pvals')
        
        return pvals < alpha/len(pvals)
    
    @staticmethod
    def sidak(pvals: pd.Series, alpha: float=.05) -> pd.Series:
        ''' Sidak correction for p-values
        
            Parameters
            ----------
            pvals: pandas.Series
                Vector of p-values to transform
                
            alpha: float, default 0.05
                Level of significance for rejecting the null hypothesis
                
            Returns
            -------
            pandas.Series:
                Whether the null hypothesis is rejected for the corresponding
                p-value
        '''
        
        guards.not_series(pvals, 'pvals')
        
        return pvals < 1 - (1 - alpha)**(1/len(pvals))
    
    @staticmethod
    def holm_bonferroni(pvals: pd.Series, alpha: float=.05) -> pd.Series:
        ''' Holm-Bonferroni correction for p-values
        
            Parameters
            ----------
            pvals: pandas.Series
                Vector of p-values to transform
                
            alpha: float, default 0.05
                Level of significance for rejecting the null hypothesis
                
            Returns
            -------
            pandas.Series:
                Whether the null hypothesis is rejected for the corresponding
                p-value
        '''
        
        guards.not_series(pvals, 'pvals')
        
        vals = pvals.sort_values().reset_index(drop=True)
        vals = pd.concat([vals, vals.index.to_series()], axis=1)
        
        return vals[0] < alpha/(len(pvals) - vals[1])
    
    @staticmethod
    def holm_sidak(pvals: pd.Series, alpha: float=.05) -> pd.Series:
        ''' Holm-Sidak correction for p-values
        
            Parameters
            ----------
            pvals: pandas.Series
                Vector of p-values to transform
                
            alpha: float, default 0.05
                Level of significance for rejecting the null hypothesis
                
            Returns
            -------
            pandas.Series:
                Whether the null hypothesis is rejected for the corresponding
                p-value
        '''
        
        guards.not_series(pvals, 'pvals')
        
        vals = pvals.sort_values().reset_index(drop=True)
        vals = pd.concat([vals, vals.index.to_series()], axis=1)
        
        return vals[0] < 1 - (1 - alpha)**(1/(len(pvals)-vals[1]))
        
    @staticmethod
    def scale(pvals: pd.Series) -> pd.Series:
        ''' Returns the negative of the p-values in a log 10 scale
        
            Parameters
            ----------
            pvals: pandas.Series
                Vector of p-values to transform
                
            Returns
            -------
            pandas.Series:
                Transformed p-values
        '''
        
        guards.not_series(pvals, 'pvals')
        
        return -np.log10(pvals)
