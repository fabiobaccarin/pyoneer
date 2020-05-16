#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Validation module
'''


import pandas as pd
import numpy as np
from pyoneer import utils, metrics, guards
from sklearn.metrics import confusion_matrix


def _printer(results: pd.DataFrame) -> None:
    ''' Print cross-validation results
    
        Parameters
        ----------
        results: pandas.DataFrame
            A dataframe containing cross-validation results to be printed
    '''

    guards.not_dataframe(results, 'results')

    print('\nCROSS VALIDATION RESULTS\n')

    columns_names = list(results)
    names = '{:^15}' * (len(columns_names) + 1)
    names = names.format('Fold', *list(columns_names))
    values = '{:^15}' + '{:^ 15.2f}' * len(columns_names)

    print('\n' + names)
    print('=' * len(names))

    for i in range(len(results)):
        print(values.format(i + 1, *results.loc[[i]].values.tolist()[0]))
    print('=' * len(names))

    mins = ['Min']
    maxs = ['Max']
    means = ['Mean']
    stds = ['Stdev']
    for column in columns_names:
        mins.append(results[column].min())
        maxs.append(results[column].max())
        means.append(results[column].mean())
        stds.append(results[column].std())
    print(values.format(*mins))
    print(values.format(*maxs))
    print(values.format(*means))
    print(values.format(*stds))
    print('=' * len(names))


def make_folds(dataframe: pd.DataFrame,
               n_folds: int=10, seed: int=None) -> pd.DataFrame:
    ''' Creates folds for cross-validation
        
        Parameters
        ----------
        dataframe: pandas.DataFrame
            Dataframe containing the data to train and test the a machine
            learning algorithm
            
        n_folds: int, default 10
            Number of subsamples (aka folds) to be used during cross-validation
            
        seed: int, default None
            Seed for random methods. Set a value to ensure reproducibility
            
        Returns
        -------
        result_df: pandas.DataFrame
            Dataframe containing all previous information, plus a column with
            the folder ID for all observations (rows)
    '''

    guards.not_dataframe(dataframe, 'dataframe')

    result_df = dataframe.copy()
    result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    folds = pd.qcut(result_df.index.tolist(), n_folds, labels=range(1, n_folds+1))
    result_df['Fold'] = np.array(folds)

    return result_df


def split(dataframe: pd.DataFrame, fold_number: int) -> pd.DataFrame:
    ''' Splits the dataframe into train and test sets. By convetion, the 
        training set is the fold equal to `fold_number`
        
        Parameters
        ----------
        dataframe: pandas.DataFrame
            Dataframe containing the information to train and test an machine
            learning algorithm
            
        fold_number: int
            Number indicating the fold (aka subsample) used as the training set
            
        Returns
        -------
        train_fold: pandas.DataFrame
            Train set for specified `fold_number`
            
        test_fold: pandas.DataFrame
            Test set for specified `fold_number`
    '''
    
    guards.not_dataframe(dataframe, 'dataframe')

    train_fold = dataframe.loc[dataframe['Fold'] == fold_number]
    train_fold.drop('Fold', axis=1)
    test_fold = dataframe.loc[dataframe['Fold'] != fold_number]
    test_fold.drop('Fold', axis=1)

    return train_fold, test_fold


def cross_validation(model, dataframe: pd.DataFrame, features_names: list,
                     target_name: str, n_folds: int=10, seed: int=None,
                     fold_step: bool=True) -> pd.DataFrame:
    ''' Performs cross-validation
        
        Parameters
        ----------
        model
            Any machine learning algorithm that has a `fit`, a `predict`, and
            a `predict_proba` method
            
        dataframe: pandas.DataFrame
            Dataframe containing the data to train and test the machine learning
            algorithm
            
        features_names: list
            List of features names as strings
            
        target_name: str
            Name of the target variable
            
        n_folds: int, default 10
            Number of folds (aka subsamples) used during cross-validation
            
        seed: int, default None
            Seed for random methods. Set a value to ensure reproducibility
            
        fold_step: bool, default True
            Flag to indicate if the there is need to partition the dataframe
            into folds (aka subsamples)
            
        Returns
        -------
        results_df: pandas.DataFrame
            Dataframe containing the results of cross-validation, namely
            the values of the metrics supported, by fold
    '''

    guards.not_dataframe(dataframe, 'dataframe')

    df = dataframe.copy()
    results = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1_score': [],
        'KS': [],
        'AUC': []
    }

    if fold_step:
        df = make_folds(df, n_folds, seed)

    for i in range(1, n_folds+1):
        train_fold, test_fold = split(df, i)
        X_train, X_test = train_fold[features_names], test_fold[features_names]
        y_train, y_test = train_fold[target_name], test_fold[target_name]
        model.fit(X_train, y_train)

        ypred = model.predict(X_test)
        cm = confusion_matrix(y_test, ypred)
        results['Accuracy'].append(metrics.accuracy(cm))
        results['Precision'].append(metrics.precision(cm))
        results['Recall'].append(metrics.recall(cm))
        results['F1_score'].append(metrics.f1score(cm))
        results['KS'].append(metrics.ks(model.predict_proba, X_test, y_test))
        results['AUC'].append(metrics.auc(model.predict_proba, X_test, y_test))

    results_df = pd.DataFrame.from_dict(results)

    _printer(results_df)

    return results_df


def cross_populate(model, dataframe: pd.DataFrame, features_names: list,
                   target_name: str, n_folds: int=10, quantiles: int=10,
                   seed: int=None, fold_step: bool=True,
                   gscore: bool=True) -> pd.DataFrame:
    ''' Populates a dataframe using predictions derived from cross-validation
    
        Parameters
        ----------
        model
            Any machine learning algorithm that has a `fit`, a `predict`, and
            a `predict_proba` method
            
        dataframe: pandas.DataFrame
            Dataframe containing the data to train and test the machine learning
            algorithm
            
        features_names: list
            List of features names as strings
            
        target_name: str
            Name of the target variable
            
        n_folds: int, default 10
            Number of folds (aka subsamples) used during cross-validation
            
        quantiles: int, default 10
            Number of quantiles to partition the predictions for future
            analysis
            
        seed: int, default None
            Seed for random methods. Set a value to ensure reproducibility
            
        fold_step: bool, default True
            Flag to indicate if the there is need to partition the dataframe
            into folds (aka subsamples)
            
        gscore: bool, default True
            Flag to indicate if the desired predicted class is encoded by zero.
            Follows credit industry conventions of calling this class the 'good'
            class
            
        Returns
        -------
        result_df: pandas.DataFrame
            Dataframe containing the all previous information and a column of
            predicted values, together with its partition in quantiles
    '''

    guards.not_dataframe(dataframe, 'dataframe')
    
    df = dataframe.copy()
    if fold_step:
        df = make_folds(df, n_folds, seed)

    result_df = pd.DataFrame()
    for i in range(1, n_folds+1):
        test_fold, train_fold = split(df, i)
        X_train, X_test = train_fold[features_names], test_fold[features_names]
        y_train = train_fold[target_name]
        model.fit(X_train, y_train)

        test_fold['SCORE'] = utils.make_score(model.predict_proba, X_test, goodscore=gscore)
        test_fold['FX_SCORE'] = pd.qcut(test_fold['SCORE'], quantiles, labels=range(1, quantiles+1))
        test_fold['FX_SCORE_DESC'] = pd.qcut(test_fold['SCORE'], quantiles)
        result_df = pd.concat([result_df, test_fold], ignore_index=True)

    return result_df
