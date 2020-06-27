'''
Type aliases to make typing less verbose
'''

import typing as t
import pandas as pd
import numpy as np

Matrix = t.Union[pd.DataFrame, np.ndarray]
Vector = t.Union[pd.Series, np.array, Matrix]
