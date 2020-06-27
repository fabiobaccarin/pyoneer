#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for the package
"""

import typing as t
import numpy as np
import pandas as pd
from pyoneer import guards


class CorrBasedSelectorMixin:
    ''' Base class for correlation-based feature selection
    
        Attributes
        ----------
        corr_cutoff: float, default 0.75
            Value above which two features should be considered pathologically
            correlated (i.e. redundant)
            
        pval_cutoff: float, default 1.0
            Value above which the p-value associated with the correlation
            measure should be considered as evidence in favor of the null
            hypothesis (there is no correlation). Note that this cut-off is
            not a statistical significance test. It is intended to be used
            as a measure of uncertainty regarding the estimated value of its
            associated correlation coefficient
            
        too_good_to_be_true: float, default 0.99
            Value above which all correlation should be considered too good
            to be true. It is used when ranking features based on a target
            (aka dependent variable or response). This parameter calibrates
            for considering features "too good to be true" for predicting
            the target
    '''
    
    def __init__(self, corr_cutoff: t.Optional[float]=.75,
            pval_cutoff: t.Optional[float]=1.0,
            too_good_to_be_true: t.Optional[float]=.99):
        self.corr_cutoff = float(corr_cutoff)
        self.pval_cutoff = float(pval_cutoff)
        self.too_good_to_be_true = float(too_good_to_be_true)

    def rank(self, X: pd.DataFrame, y: pd.Series,
            apply_cutoffs: t.Optional[bool]=True) -> pd.DataFrame:
        ''' Ranks attributes (columns) in X according to its correlation with
            the target y
            
            Parameters
            ----------
            X: pandas.DataFrame
                Matrix of attributes whose columns are to be ranked according
                to their correlation with y
                
            y: pandas.Series
                Vector of response measures (aka dependent variable or target)
                to use for ranking features
                
            apply_cutoffs: bool, default True
                Whether to apply the cut-offs associated if the object during
                ranking. See the documentation on the class attributes for
                more information on these
                
            Returns
            -------
            df: pandas.DataFrame
                Dataframe containing 4 columns and n rows, where rows is at
                most the number of columns in X. The first column is the
                association measure (`assoc`), the second and third its
                associated p-value and the fourth is the ranking of the
                variable, where the number 1 indicates the best one, the
                number 2 the second-best and so on
        '''
        
        guards.not_dataframe(X, 'X')
        guards.not_series(y, 'y')
        
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
    
    @staticmethod
    def filter(X: pd.DataFrame, corr: pd.DataFrame,
               cutoff: float, ranking: t.Union[t.List[str], t.Tuple[str, ...]]
               ) -> pd.DataFrame:
        ''' Applies the correlation cut-off to X, according to the correlations
            calculated in the correlation matrix corr. It uses a ordered list
            of feature names to start the cutting from the most important
            variables according to the ranking method. In this list, the first
            element is the most important variable (ranked 1), the second
            the second most important variable (ranked 2) and so on
            
            Parameters
            ----------
            X: pandas.DataFrame
                Matrix of attributes whose columns are to be filtered according
                to their correlation with each other
                
            corr: pandas.DataFrame
                Correlation matrix used for assessing each pairwise correlation
                of the columns of X. It should be a square matrix with dimension
                n x n, where n is the number of columns in X. Preferably, it
                should be symmetric and have only number 1 on its main diagonal.
                For an example of this matrix, see the result of
                pandas.DataFrame.corr method
                
            cutoff: float
                Correlation above which the pair of variables are considered
                to be pathologically correlated
                
            ranking: Union[List[str], Tuple[str, ...]]
                List or tuple of variable names that will be used as a ranking. 
                When doing the filtering, this method will always drop the 
                variable in the pair that comes AFTER the other one in this 
                list. This means that the first item of this list should be the 
                most important variable (ranked 1), the second should be the
                second most important variable (ranked 2) and so on
                
            Returns
            -------
            pandas.DataFrame:
                Dataframe with the columns in X which were deemed not
                pathologically correlated with any of the others. It has
                the same number of rows as X
        '''
        
        guards.not_dataframe(X, 'X')
        guards.not_dataframe(corr, 'corr')
        guards.not_iterable(ranking, 'ranking')
        
        keep = set(ranking)
        for i in ranking:
            keep -= set([j for j in ranking
                         if ranking.index(i) < ranking.index(j)
                         and corr.at[i, j] > cutoff])
        return X[keep]
    
    def _stop(self, X: pd.DataFrame) -> bool:
        ''' Stop condition of the fit algorithm
        
            Parameters
            ----------
            X: pandas.DataFrame
                Matrix of attributes being evaluated
                
            Returns
            -------
            bool:
                Whether the maximum absolute pairwise correlation in X is
                less than the correlation cutoff specified when instantiating
                the object
        '''
        
        guards.not_dataframe(X, 'X')
        
        return self.corr(X).abs().max().max() < self.corr_cutoff
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            apply_cutoffs: t.Optional[bool]=True,
            ranking: t.Optional[t.Union[t.List[str], t.Tuple[str, ...]]]=None
            ) -> pd.DataFrame:
        ''' Applies correlation-based feature selection for a attribute
            matrix X and a response vector y
            
            Parameters
            ----------
            X: pandas.DataFrame
                Matrix of attributes whose columns are to be ranked according
                to their correlation with y
                
            y: pandas.Series
                Vector of response measures (aka dependent variable or target)
                to use for ranking features
                
            apply_cutoffs: bool, default True
                Whether to apply the cut-offs associated if the object during
                ranking. See the documentation on the class attributes for
                more information on these
            
            ranking: Union[List[str], Tuple[str, ...]], default None
                List or tuple of variable names that will be used as a ranking. 
                When doing the filtering, this method will always drop the 
                variable in the pair that comes AFTER the other one in this 
                list. This means that the first item of this list should be the 
                most important variable (ranked 1), the second should be the
                second most important variable (ranked 2) and so on. If no
                such list is provided, one is created using the `rank` method
                of this class
                
            Returns
            -------
            X_new: pandas.DataFrame
                Dataframe with the columns in X which were deemed not
                pathologically correlated with any of the others. It has
                the same number of rows as X
        '''
        
        guards.not_dataframe(X, 'X')
        guards.not_series(y, 'y')
        guards.not_iterable(ranking, 'ranking')
        
        if ranking is None:
            rk = (self.rank(X, y, apply_cutoffs)['rank'].sort_values()
                  .index.to_list())
        else:
            rk = ranking
        corr = self.corr(X).abs()
        for r in np.sort(np.linspace(0, self.corr_cutoff))[::-1]:
            X_new = self.filter(X, corr, r, rk)
            if self._stop(X_new):
                return X_new
