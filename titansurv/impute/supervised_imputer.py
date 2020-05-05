from sklearn.base import BaseEstimator

# TODO: Add support for lower and upper bounds?
class SupervisedImputer(BaseEstimator):
    '''Perform supervised imputation using any estimator
    
    Note: Currently only one column imputation is supported
    
    Parameters
    ----------
    
    estimator: an estimator object
    
    key_NA: str
        the column to impute provided as a string
        
    keys_nonNA: list
        the regressors for the imputation provided as a list of column keys
        
    df_out: bool default=False
        whether to return a dataframe or not
    '''
    
    def __init__(self ,estimator, key_NA, keys_nonNA=None, df_out=False):
        self.estimator = estimator
        self.key_NA = key_NA
        self.keys_nonNA = keys_nonNA
        self._has_no_NA = False
        self.df_out = df_out
        
        
    def fit(self, X, y=None, *args, **kwargs):
        
        X = X.dropna(subset=[self.key_NA])
        y = X.loc[:, self.key_NA]
        
        if self.keys_nonNA is None:
            X = X.drop(self.key_NA, axis=1)
        else:
            X = X.loc[:, self.keys_nonNA]
                    
        self.estimator.fit(X, y)
        
        return self
        
        
    def transform(self, X, y=None, *args, **kwargs):
        y = X.loc[:, self.key_NA]
        y_is_NA = y.isna()
        
        if not y_is_NA.any():
            return X        
        
        X =  X.copy()
        X_NA = X.loc[y_is_NA, :].drop(self.key_NA, axis=1)
        X.loc[y_is_NA, self.key_NA] = self.estimator.predict(X_NA)
        
        if not self.df_out:
            return X.values
        
        return X
    
    def fit_transform(self, X, y=None, *args, **kwargs):
        return self.fit(X, y, *args, **kwargs).transform(X, y, *args, **kwargs)