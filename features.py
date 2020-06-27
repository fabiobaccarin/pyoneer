"""
Module for feature processing. Includes feature selection and feature
engineering algorithms
"""

import typing as t
import numpy as np
import pandas as pd
import warnings
from pyoneer import guards, metrics
from pyoneer.base import CorrBasedSelectorMixin
from pyoneer import warn_messages as w
from scipy import stats as ss


class OptimalMeasureSelector(CorrBasedSelectorMixin):
    ''' Optimal measure correlation-based feature selection class. "Optimal"
        means that the measure chosen as a correlation coefficient depends
        on the type of the variables in the pair. Essentially, this class
        treats pairs of categorical variables differently from the other
        three pairs possible (continuous-continuous, continuous-categorical
        and categorical-continuous). If the pair correspond to only categorical
        variables, the measure chosen as correlation coefficient is the
        Yule's Q (see the `metrics` module for more details). For the other
        cases, the measure is the Spearman correlation coefficient
    
        Attributes
        ----------
        catvars: Union[List[str], Tuple[str, ...]]
            List of attribute names which are to be considered categorical
            variables. Every variable that is not in this list will be deemed
            continuous
        
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
    
    def __init__(self, catvars: t.Union[t.List[str], t.Tuple[str, ...]], 
            corr_cutoff: t.Optional[float]=.75,
            pval_cutoff: t.Optional[float]=1.0,
            too_good_to_be_true: t.Optional[float]=.99):
        self.catvars = catvars
        super().__init__(corr_cutoff, pval_cutoff, too_good_to_be_true)
        
    def _catcat(self, x: str, y: str) -> bool:
        ''' Returns `True` if both variables x and y must be considered
            categorical
            
            Parameters
            ----------
            x: str
                Name of the first variable in the pair
            
            y: str
                Name of the second variable in the pair
                
            Returns
            -------
            bool:
                Whether x and y are both to be considered categorical variables
        '''
        
        return x in self.catvars and y in self.catvars
    
    def assoc(self, var1: pd.Series, var2: pd.Series) -> (float, float):
        ''' Calculates the optimal correlation measure for the pair of
            variables var1 and var2, according to the rules on the class
            description
            
            Parameters
            ----------
            var1: pandas.Series
                Vector of values of the first variable
                
            var2: pandas.Series
                Vector of values of the second variable
                
            Returns
            -------
            r: float
                The value of the correlation coefficient estimated
                
            pval: float
                The p-value associated with the estimative. The null hypothesis
                considered is that the correlation is zero
        '''
        
        guards.not_series(var1, 'var1')
        guards.not_series(var2, 'var2')
        
        try:
            if self._catcat(var1.name, var2.name):
                odds_ratio, pval = ss.fisher_exact(pd.crosstab(var1, var2))
                r = metrics.yule_q(odds_ratio)        
            else:
                r, pval = ss.spearmanr(var1, var2)
            return r, pval
        except ValueError:
            warnings.warn(
                w.ASSOC_WARNING.format(var1.name, var2.name),
                RuntimeWarning
            )
            return np.nan, np.nan
    
    def corr(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Calculates a correlation matrix for df with the method on the
            class description. It mimics the behavior of `pandas.DataFrame.corr`
            method
            
            Parameters
            ----------
            df: pandas.DataFrame
                Matrix for which to calculate all pairwise correlations
                
            Returns
            -------
            matrix: pandas.DataFrame
                Square matrix corresponding to the correlation matrix of
                every variable (column) in df. It is a symmetric matrix, with
                only 1 on its main diagonal
        '''
        
        guards.not_dataframe(df, 'df')
        
        cols = df.columns.to_list()
        matrix = pd.DataFrame({k: np.nan for k in cols}, index=cols)
        for i in cols:
            matrix.at[i, i] = 1
            for j in cols:
                if cols.index(i) < cols.index(j):
                    r = self.assoc(df[i], df[j])[0]
                    matrix.at[i, j] = r
                    matrix.at[j, i] = r
        return matrix
    
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
        
        return super().rank(X, y, apply_cutoffs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            apply_cutoffs: t.Optional[bool]=True,
            ranking: t.Union[t.List[str], t.Tuple[str, ...]]=None
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
        
        return super().fit(X, y, apply_cutoffs, ranking)


class SpearmanCorrSelector(CorrBasedSelectorMixin):
    ''' Correlation-based feature selection using Spearman's correlation
        coefficient
    
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
        super().__init__(corr_cutoff, pval_cutoff, too_good_to_be_true)
    
    def assoc(self, var1: pd.Series, var2: pd.Series) -> (float, float):
        ''' Calculates Spearman's correlation coefficient for the pair
            of variables var1 and var2
            
            Parameters
            ----------
            var1: pandas.Series
                Vector of values of the first variable
                
            var2: pandas.Series
                Vector of values of the second variable
                
            Returns
            -------
            r: float
                The value of the correlation coefficient estimated
                
            pval: float
                The p-value associated with the estimative. The null hypothesis
                considered is that the correlation is zero. As stated in the
                Scipy documentation, this value should only be trusted
                for sample sizes of at least 500 observations
        '''
        
        return ss.spearmanr(var1, var2)
    
    def corr(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Calculates a correlation matrix for df with the method on the
            class description. It mimics the behavior of `pandas.DataFrame.corr`
            method
            
            Parameters
            ----------
            df: pandas.DataFrame
                Matrix for which to calculate all pairwise correlations
                
            Returns
            -------
            matrix: pandas.DataFrame
                Square matrix corresponding to the correlation matrix of
                every variable (column) in df. It is a symmetric matrix, with
                only 1 on its main diagonal
        '''
        
        return df.corr('spearman')

    def rank(self, X: pd.DataFrame, y: pd.Series,
             apply_cutoffs: bool=True) -> pd.DataFrame:
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
        
        return super().rank(X, y, apply_cutoffs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            apply_cutoffs: t.Optional[bool]=True,
            ranking: t.Union[t.List[str], t.Tuple[str, ...]]=None
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
        
        return super().fit(X, y, apply_cutoffs, ranking)


class VIFSelector(CorrBasedSelectorMixin):
    ''' Selection of features based on the Variance Inflation Factor (VIF).
        For more details, se the metrics module
        
        Attributes
        ----------
        vif_cutoff: float, default 5.0
            VIF value above which the variable is considered multicollinear
        
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
    
    def __init__(self, vif_cutoff: t.Optional[float]=5.0,
            corr_cutoff: t.Optional[float]=.75,
            pval_cutoff: t.Optional[float]=1.0,
            too_good_to_be_true: t.Optional[float]=.99):
        self.vif_cutoff = vif_cutoff
        super().__init__(corr_cutoff, pval_cutoff, too_good_to_be_true)
        
    def assoc(self, var1: pd.Series, var2: pd.Series) -> (float, float):
        ''' Calculates Pearson's correlation coefficient for the pair
            of variables var1 and var2
            
            Parameters
            ----------
            var1: pandas.Series
                Vector of values of the first variable
                
            var2: pandas.Series
                Vector of values of the second variable
                
            Returns
            -------
            r: float
                The value of the correlation coefficient estimated
                
            pval: float
                The p-value associated with the estimative. The null hypothesis
                considered is that the correlation is zero
        '''
        
        return ss.pearsonr(var1, var2)
        
    def corr(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.corr('pearson')
    
    def rank(self, X: pd.DataFrame, y: pd.Series,
             apply_cutoffs: t.Optional[bool]=True) -> pd.DataFrame:
        ''' Ranks attributes (columns) in X according to its correlation with
            the target y. It uses Pearson's correlation coefficient
            
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
        
        return super().rank(X, y, apply_cutoffs)
    
    def _stop(self, X: pd.DataFrame) -> bool:
        ''' Stop condition of the fit algorithm
        
            Parameters
            ----------
            X: pandas.DataFrame
                Matrix of attributes being evaluated
                
            Returns
            -------
            bool:
                Whether the maximum VIF value for variables in X is
                less than the cut-off specified when instantiating the object
        '''
        
        guards.not_dataframe(X, 'X')
        
        return metrics.vif(X).max() < self.vif_cutoff
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            apply_cutoffs: t.Optional[bool]=True,
            ranking: t.Union[t.List[str], t.Tuple[str, ...]]=None
            ) -> pd.DataFrame:
        ''' Applies VIF-based feature selection for a attribute
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
        
        return super().fit(X, y, apply_cutoffs, ranking)
    

class LowVarianceSelector:
    ''' Class for identifying and systematically remove low variance
        features. It relies on two basic measures: the percentage of distinct
        values of a variable relative to the sample size (number of rows of a
        dataframe), and the ratio between the frequencies of the two most
        frequent values of a variable. A third measure, which is a combination
        of the previous two, is the low variance index. It is defined as the
        log10 of the ratio between the frequency ratio and the percentage of
        distinct values. The greater the frequency ratio, the lower the
        variance; the higher the percentange of distinct values, the lower the
        variance.
        
        Attributes
        ----------
        pct_distinct_cutoff: float, default 0.10
            Value below which the percentage of distinct values in a variable
            is considered as proof of low variance
            
        freq_ratio_cutoff: float, default 20.0
            Value above which the ratio of the two most frequent values in a
            variable is considered as proof of low variance
    '''
    
    def __init__(self, pct_distinct_cutoff: t.Optional[float]=0.1,
            freq_ratio_cutoff: t.Optional[float]=20.0):
        self.pct_distinct_cutoff = pct_distinct_cutoff
        self.freq_ratio_cutoff = freq_ratio_cutoff
        
    @staticmethod
    def pct_distinct(X: pd.DataFrame) -> pd.Series:
        ''' Calculates the percentage of distinct values relative to the sample
            size (number of rows) for every column in X
            
            Parameters
            ----------
            X: pandas.DataFrame
                Dataframe to evaluate
                
            Returns
            -------
            pct: pandas.Series
                Series of resulting values
        '''
        
        guards.not_dataframe(X, 'X')
        
        pct = X.apply(lambda sr: len(sr.value_counts()) / X.shape[0])
        pct.name = 'pct_distinct'
        
        return pct
    
    @staticmethod
    def freq_ratio(X: pd.DataFrame) -> pd.Series:
        ''' Calculates the ratio between the two most frequent values in a
            variable for every column in X
            
            Parameters
            ----------
            X: pandas.DataFrame
                Dataframe to evaluate
                
            Returns
            -------
            pandas.Series
                Series of resulting values
        '''
        
        guards.not_dataframe(X, 'X')
        
        freq_ratio_ = {}
        for var in X.columns.to_list():
            vcount = X[var].value_counts()
            freq_ratio_[var] = vcount.iloc[0] / vcount.iloc[1]
            
        return pd.Series(freq_ratio_, name='freq_ratio')
        
    def rank(self, X: pd.DataFrame, apply_cutoffs: t.Optional[bool]=False
            ) -> pd.DataFrame:
        ''' Creates a ranking of low variance variables, from the ones with
            least variance to the ones with most variance. Therefore, the
            variable with rank equals to 1 is the variable with least variance
            among those in X
            
            Parameters
            ----------
            X: pandas.DataFrame
                Attribute matrix
                
            apply_cutoffs: bool, default False
                Whether to apply the cut-offs specified during the
                inicialization of the object. It is recommended not to apply
                the cut-offs without first evaluating the low variance index
                (LVI), because variables with few values can be still be useful
                as categorical features
                
            Returns
            -------
            df: pandas.DataFrame
                Dataframe containing 3 columns and n <= k rows, where k is
                the number of columns in X. The first columns is the percentage
                of distinct values relative to the sample size (`pct_distinct`).
                The second column is the ratio between the frequencies of the
                two most frequent values (`freq_ratio`). The third is the
                low variance index (LVI), which is the logaritm (base 10) of
                the ratio between `freq_ratio` and `pct_distinct`
        '''
        
        guards.not_dataframe(X, 'X')
        
        df = pd.concat([self.pct_distinct(X), self.freq_ratio(X)], axis=1)
        
        if apply_cutoffs:
            pct_filter = df['pct_distinct'] < self.pct_distinct_cutoff
            freq_filter = df['freq_ratio'] > self.freq_ratio_cutoff
            df = df[pct_filter & freq_filter]
        
        df['lvi'] = np.log10(df['freq_ratio'] / df['pct_distinct'])
        df['rank'] = df['lvi'].rank(ascending=False, method='dense')
        
        return df
        
def nan_pct(df: pd.DataFrame, ascending: t.Optional[bool]=False):
    ''' Shows feature names and their corresponding percentage of missing
        values

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe with variable values

        ascending: bool, default False
            Whether to sort values in ascending order of percentage of NaNs

        Returns
        -------
        pandas.Series:
            Series of variable names and their corresponding percentage of
            NaNs
    '''

    guards.not_dataframe(df, 'df')

    return (df[[var for var in df.columns if df[var].isna().sum() > 0]].isna()
        .mean().sort_values(ascending=ascending))
