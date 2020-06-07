#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:43:38 2020

@author: Fabio Baccarin

Guards
"""

import pandas as pd
from collections import abc
from pyoneer import errors


def not_callable(obj, name) -> None:
    ''' Raises a `TypeError` if `obj` is not a callable object
        (e.g. a function)
    '''
    
    if not isinstance(obj, abc.Callable):
        raise TypeError(errors.NOT_CALLABLE.format(name))
        

def not_dataframe(obj, name) -> None:
    ''' Raises a `TypeError` if the object is not a `pandas.DataFrame` '''
    
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(errors.NOT_A_DATAFRAME.format(name))
        

def not_series(obj, name) -> None:
    ''' Raises a `Type Error` if `obj` is not a pandas.Series '''
    
    if not isinstance(obj, pd.Series):
        raise TypeError(errors.NOT_A_SERIES.format(name))
        

def not_both_none(val1, val2, names) -> None:
    ''' Raises a `ValueError` if `val1` and `val2` are both `None` '''
    
    if val1 is None and val2 is None:
        raise ValueError(errors.NOT_BOTH_NONE.format(*names))
        

def not_in_supported_values(val, supported) -> None:
    ''' Raises a `ValueError` if `val` is not in the collection of supported
        values
    '''
    
    if val not in supported:
        raise ValueError(errors.NOT_IN_SUPPORTED_VALUES.format(val, supported))
        
        
def is_none(val, name) -> None:
    ''' Raises a `ValueError` if `val` is None '''
    
    if val is None:
        raise ValueError(errors.IS_NONE.format(name))
        
def not_iterable(obj, name) -> None:
    ''' Raise a `TypeError` if `obj` is not iterable'''
    
    if not isinstance(obj, abc.Iterable):
        raise TypeError(errors.NOT_ITERABLE.format(name))

        
def not_int(obj, name) -> None:
    if not isinstance(obj, int):
        raise TypeError(errors.NOT_INT.format(name))