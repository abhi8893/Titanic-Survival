from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone, BaseEstimator, TransformerMixin


# implement this as an abstract base class?
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
        

class ColumnDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        return X.drop(self.key, axis=1)


# TODO: Can I implement this to support both dataframes and arrays?
# TODO: Implement key='auto' to drop all NaNs
class NaNDropper(BaseEstimator, AutoFitTrans):
    
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

    
class SimpleImputerDF(BaseEstimator, TransformerMixin, AutoFitTrans):
    
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



class Passthrough(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
