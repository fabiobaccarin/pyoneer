#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guards
"""

import typing as t
import pandas as pd
from collections import abc
from pyoneer import errors


def not_callable(obj: t.Any, name: str) -> None:
    ''' Raises a `TypeError` if `obj` is not a callable object
        (e.g. a function)
    '''
    
    if not isinstance(obj, abc.Callable):
        raise TypeError(errors.NOT_CALLABLE.format(name))
        

def not_dataframe(obj: t.Any, name: str) -> None:
    ''' Raises a `TypeError` if the object is not a `pandas.DataFrame` '''
    
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(errors.NOT_A_DATAFRAME.format(name))
        

def not_series(obj: t.Any, name: str) -> None:
    ''' Raises a `Type Error` if `obj` is not a pandas.Series '''
    
    if not isinstance(obj, pd.Series):
        raise TypeError(errors.NOT_A_SERIES.format(name))
        

def not_both_none(val1: t.Any, val2: t.Any,
        names: t.Union[t.List[str], t.Tuple[str, ...]]) -> None:
    ''' Raises a `ValueError` if `val1` and `val2` are both `None` '''
    
    if val1 is None and val2 is None:
        raise ValueError(errors.NOT_BOTH_NONE.format(*names))
        

def not_in_supported_values(val: t.Any,
        supported: t.Union[t.List[t.Any], t.Tuple[t.Any, ...]]) -> None:
    ''' Raises a `ValueError` if `val` is not in the collection of supported
        values
    '''
    
    if val not in supported:
        raise ValueError(errors.NOT_IN_SUPPORTED_VALUES.format(val, supported))
        
        
def is_none(val: t.Any, name: str) -> None:
    ''' Raises a `ValueError` if `val` is None '''
    
    if val is None:
        raise ValueError(errors.IS_NONE.format(name))


def not_iterable(obj: t.Any, name: str) -> None:
    ''' Raise a `TypeError` if `obj` is not iterable'''
    
    if not isinstance(obj, abc.Iterable):
        raise TypeError(errors.NOT_ITERABLE.format(name))

        
def not_int(obj: t.Any, name: str) -> None:
    if not isinstance(obj, int):
        raise TypeError(errors.NOT_INT.format(name))
