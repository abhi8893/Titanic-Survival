from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone


class AutoFitTrans:
    '''
    Use this to implement fit_transform automatically
    '''
    
    def fit(self):
        pass
    
    def transform(self):
        pass
    
    def fit_transform(self, *args, **kwargs):
        return self.fit(*args, **kwargs).transform(*args, **kwargs)
        

class ColumnDropper(BaseEstimator, ClassifierMixin):
    
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        return X.drop(self.key, axis=1)


# TODO: Can I implement this to support both dataframes and arrays?
# TODO: Implement key='auto' to drop all NaNs
class NaNDropper(BaseEstimator, ClassifierMixin, AutoFitTrans):
    
    '''Drops rows with NaN values
    
    key: list-like
        A list of keys(column names) to consider while dropping NaN values
    '''
    
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y=None):
        '''Fits the model and extracts indices with missing values
        
        Parameters
        ----------
        
        X: pd.DataFrame
        y: pd.Series (Default: None)
        '''
        self.nan_indices = X.loc[:, self.key].isna().any(axis=1)
        if y is not None:
            self.nan_indices = self.nan_indices | y.isna()


        return self
    
    def transform(self, X, y=None):
        if y is None:
            return X.loc[~self.nan_indices]
        else:
            return X.loc[~self.nan_indices], y.loc[~self.nan_indices]
        
#     def fit_transform(self, X, y=None):
#         return self.fit(X, y).transform(X, y)

def __convert_to_list(obj):
    if not isinstance(obj, list):
        return [obj]




def modify_transformer_cols(col_trnsfrmr: ColumnTransformer, append=False, inplace=False, **trnsfrmr_cols):
    if not inplace:
        new_col_trnsfrmr = clone(col_trnsfrmr)

    trnsfrmrs = new_col_trnsfrmr.transformers
    for i, [trnsfrmr_name, old_trnsfrmr, old_cols] in enumerate(trnsfrmrs):
        new_cols = trnsfrmr_cols.get(trnsfrmr_name, None)
        
        if new_cols is not None:
            if append:
                new_cols  = list(set().union(new_cols, old_cols))
        else:
            new_cols = old_cols
                            
        trnsfrmrs[i] = (trnsfrmr_name, old_trnsfrmr, new_cols)
        
    return new_col_trnsfrmr


def modify_transformer_est(col_trnsfrmr: ColumnTransformer, inplace=False, **trnsfrmr_estmtrs):
    if not inplace:
        new_col_trnsfrmr = clone(col_trnsfrmr)

    trnsfrmrs = new_col_trnsfrmr.transformers
    for i, [trnsfrmr_name, old_trnsfrmr, old_cols] in enumerate(trnsfrmrs):
        new_trnsfrmr = trnsfrmr_estmtrs.get(trnsfrmr_name, old_trnsfrmr)

        trnsfrmrs[i] = (trnsfrmr_name, new_trnsfrmr, old_cols)


    return new_col_trnsfrmr


def remove_transformer_by_name(col_trnsfrmr: ColumnTransformer, names: list, inplace=False):
    if not inplace:
        new_col_trnsfrmr = clone(col_trnsfrmr)

    trnsfrmrs = new_col_trnsfrmr.transformers
    for i, [trnsfrmr_name, old_trnsfrmr, old_cols] in enumerate(trnsfrmrs):
        if trnsfrmr_name in names:
            trnsfrmrs.pop(i)

    return new_col_trnsfrmr
    



    
class SimpleImputerDF(BaseEstimator, ClassifierMixin, AutoFitTrans):
    
    def __init__(self, keys, strategy='mean', fill_value=None):
        self.keys = keys
        self.strategy = strategy
        self.fill_value = fill_value
        
    
    def fit(self, X, y=None):
        if self.fill_value is not None:
            self.fill_value = X.loc[:, self.keys].agg(self.strategy)
            
        return self
            
            
    def transform(self, X, y=None):
        sel_cols = list(set(X.columns) & set(self.keys))
        X.loc[:, sel_cols] = X.loc[:, sel_cols].apply(lambda c: c.fillna(self.fill_value[c.name]), axis=0)
        return X



class Passthrough(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)