"""
Guards
"""

import typing as t
import pandas as pd
from collections import abc
from pyoneer import errors as e


def not_callable(obj: t.Any, name: str) -> None:
    if not isinstance(obj, abc.Callable):
        raise e.NotCallableError(e.NotCallableError.message.format(name))
        

def not_dataframe(obj: t.Any, name: str) -> None:
    if not isinstance(obj, pd.DataFrame):
        raise e.NotDataFrameError(e.NotDataFrameError.message.format(name))
        

def not_series(obj: t.Any, name: str) -> None:
    if not isinstance(obj, pd.Series):
        raise e.NotSeriesError(e.NotSeriesError.message.format(name))
        

def not_both_none(val1: t.Any, val2: t.Any,
        names: t.Union[t.List[str], t.Tuple[str, ...]]) -> None:
    if val1 is None and val2 is None:
        raise e.BothNoneError(e.BothNoneError.message.format(*names))
        

def not_in_supported_values(val: t.Any,
        supported: t.Union[t.List[t.Any], t.Tuple[t.Any, ...]]) -> None:
    if val not in supported:
        raise e.NotInSupportedValuesError(
            e.NotInSupportedValuesError.message.format(val, supported)
        )
        
        
def is_none(val: t.Any, name: str) -> None:
    if val is None:
        raise e.IsNoneError(e.IsNoneError.message.format(name))


def not_iterable(obj: t.Any, name: str) -> None:
    if not isinstance(obj, abc.Iterable):
        raise e.NotIterableError(e.NotIterableError.message.format(name))

        
def not_int(obj: t.Any, name: str) -> None:
    if not isinstance(obj, int):
        raise e.NotIntegerError(e.NotIntegerError.message.format(name))
