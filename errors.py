'''
Errors
'''

class BaseError(Exception):
    pass


class NotCallableError(BaseError):
    message = "Object '{}' is not callable (e.g. a function)"


class NotDataFrameError(BaseError):
    message = "Object '{}' is not a pandas.DataFrame"


class NotSeriesError(BaseError):
    message = "Object '{}' is not a pandas.Series"


class BothNoneError(BaseError):
    message = "You must specify a value for either '{}' or '{}'. They cannot both be None"


class NotInSupportedValuesError(BaseError):
    message = '{} is not in the list of supported values: {}'


class IsNoneError(BaseError):
    message = "You must set a value for '{}'. It cannot be None"


class NotIterableError(BaseError):
    message = "Object '{}' is not iterable"


class NotIntegerError(BaseError):
    message = "Object '{}' is not of type `int`"
