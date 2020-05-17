#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Utils module
'''

import numpy as np
import pandas as pd
from collections import abc
from pyoneer import guards


def make_score(predictor: abc.Callable, X, good_score: bool=True) -> np.array:
    ''' Returns a vector of scores in the range 0-1000 (integer)
    
        Parameters
        ----------
        predictor: abc.Callable
            Any function that uses information to make predicitions
            
        X: numpy.ndarray, pandas.Series
            Matrix of attributes to be used in making predictions
            
        good_score: bool, default True
            Flag to indicate if predictions should be for the zero class.
            Follows the credit industry convention of calling it the 'good' class
            
        Returns
        -------
        numpy.ndarray
            Vector with scores
    '''

    guards.not_callable(predictor, 'predictor')
    
    col = 0 if good_score else 1
    
    scores = predictor(X)[:, col]

    return np.array([int(round(scores[i] * 1000)) for i in range(len(X))])


def set_column_val(df: pd.DataFrame, col: str, val) -> pd.DataFrame:
    ''' Returns a new dataframe with the values in column named `col` equal
        to `val`
        
        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe whose column one wants to set a value to
            
        col: str
            Name of the column in `df` to be changed
            
        val: Any
            Value to be set in the column
            
        Returns
        -------
        df_new: pandas.DataFrame
            New dataframe equal to `df` except that all values in column `col`
            are equal to `val`
    '''
    
    guards.not_dataframe(df, 'df')
    
    df_new = df.copy()
    df_new[col] = val
    
    return df_new


def shuffle_dataframe_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ''' Returns a new dataframe with the values in column named `col`
        shuffled randomly
        
        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe to have one of its columns shuffled
            
        col: str
            Name of the column in `df` whose values need to be shuffled
            
        Returns
        -------
        df_new: pandas.DataFrame
            New dataframe, equal to `df` except that the values in column
            named `col` are shuffled
    '''
    
    guards.not_dataframe(df, 'df')
    
    df_new = df.copy()
    df_new[col] = np.random.permutation(df_new[col].values)
    
    return df_new


class DataFrameFormatter:
    ''' Contains tools for changing the format of a pandas DataFrame '''
    
    @staticmethod
    def matrix_to_long(df: pd.DataFrame, values_label: str,
                       dtype: str='float64') -> pd.DataFrame:
        ''' Returns the dataframe in a 'long' format, with the first column named
            'var' indicating the variable previously on the column, and the second
            column corresponding to the value. This column will be named
            `values_label`
        '''
        
        guards.not_dataframe(df, 'df')
        
        long = pd.Series(dtype=dtype, name=values_label)
        length = len(df)
        for var in df.columns.to_list():
            series = pd.Series(df[var].values,
                               index=[var for _ in range(length)],
                               dtype=dtype, name=values_label)
            long = pd.concat([long, series])
            
        return long.reset_index().rename(columns={'index': 'var'})
    
    @staticmethod
    def long_to_matrix(df: pd.DataFrame, var_labels_col: str, values_col: str,
                       dtype: str='float64') -> pd.DataFrame:
        ''' Returns the dataframe in 'matrix' format, with k/n rows, where
            k is the number of rows in `df` and n is the number of unique
            values in column `var_labels_col`. The number of columns in the
            output is equal to n also.
        '''
        
        guards.not_dataframe(df, 'df')
        
        cols = df[var_labels_col].unique().tolist()
        matrix = pd.DataFrame(columns=cols)
        
        for var in cols:
            matrix[var] = (df.query(f'{var_labels_col} == "{var}"')[values_col]
                           .reset_index(drop=True))
            
        return matrix