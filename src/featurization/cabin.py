from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class CabinTypeExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, grp_cat=True, enc='ohe', sparse=False):
        self.grp_cat = grp_cat
        self.enc = enc
        
    
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

        X = pd.DataFrame(X)

        if not self.enc:
            return X
            
        return self.ohe.transform(X)
    
    def get_feature_names(self, input_features):
        return self.ohe.get_feature_names(input_features)



import re
from sklearn.preprocessing import FunctionTransformer


def _get_num_cabin(x):
    return len(re.findall(r'[A-Z]\d', x))

def get_num_cabin(cabin):
    cabin = cabin.fillna('NC')
    return pd.DataFrame(cabin.apply(_get_num_cabin))


numCabin = FunctionTransformer(get_num_cabin)
    