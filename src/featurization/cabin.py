from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class CabinTypeExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, grp_cat=True, sparse=False):
        self.grp_cat = grp_cat
        
        if self.grp_cat:
            self.exclude_cat = ['A', 'B', 'C', 'D', 'E', 'NC']
            self.cat = self.exclude_cat + ['other']
            self.ohe = OneHotEncoder([self.cat], drop=['other'], sparse=sparse)
        else:
            self.cat = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'NC']
            self.ohe = OneHotEncoder([self.cat], drop=['T'], sparse=sparse)
            
        self.sparse = sparse
        
    def fit(self, X, y=None):
                    
        try:    
            name = X.name 
        except AttributeError:
            name = 'Cabin'
            
        dummy_df = pd.DataFrame({name: ['A']})
        
        self.ohe.fit(dummy_df)
        
        return self
    
    
    def transform(self, X):
        X = X.str[0].fillna('NC')
        if self.grp_cat:
            X.loc[~X.isin(self.exclude_cat)] = 'other'

        return self.ohe.transform(pd.DataFrame(X))
    
    def get_feature_names(self, input_features):
        return self.ohe.get_feature_names(input_features)

