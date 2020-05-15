from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class TicketTypeExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self,  only_numeric=False, exclude_thresh=10, drop='auto', 
                 handle_unknown='prespecify', sparse=False):
        self.exclude_thresh = exclude_thresh
        self.only_numeric = only_numeric
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.sparse = sparse
        
    @staticmethod
    def replace(v):
        if not v.isdigit():
            return v.split(' ')[0]
        else:
            return 'numeric'
        
        
    def fit(self, X, y=None):
            
        if self.only_numeric:
            self.cat = ['numeric', 'other']
        else:
            if self.exclude_thresh <= 10:
                self.exclude_cat = ['PC', 'CA', 'A5', 'SOTONOQ', 'STONO']
            elif self.exclude_thresh <= 12:
                self.exclude_cat = ['PC', 'CA', 'A5', 'SOTONOQ']
            elif self.exclude_thresh <= 15:
                self.exclude_cat = ['PC', 'CA', 'A5']
                                
            self.cat = self.exclude_cat + ['numeric', 'other']
            
        try:    
            name = X.name 
        except AttributeError:
            name = 'Ticket'
            
            
        dummy_df = pd.DataFrame({name: ['numeric']})
        
        self.ohe = OneHotEncoder(categories=[self.cat], drop=['other']).fit(dummy_df)
        
        return self
    
    def transform(self, X):
        X = X.apply(self.replace)
        
        if self.only_numeric:
            X.loc[~X.isin(['numeric'])] = 'other'

        X.loc[~X.isin(['numeric'] + self.cat)] = 'other'
    
        
        return self.ohe.fit_transform(pd.DataFrame(X))
    
    def get_feature_names(self, input_features=None):
        return self.ohe.get_feature_names(input_features)
    
    