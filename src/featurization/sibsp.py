from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd


class SibSpBinner(BaseEstimator, TransformerMixin):
    

    def __init__(self, thresh=2, encode='ord', sparse=False):
        
        if thresh > 7:
            raise ValueError('Specify a value less than 7')
            
        self.thresh = thresh
        self.cat = self.get_cat(thresh)
        self.encode = encode
        self.sparse = sparse
        
        if encode == 'ohe':
            self.enc = OneHotEncoder([self.cat], drop=[self.cat[-1]], sparse=sparse)
        elif encode == 'ord':
            self.enc = OrdinalEncoder([self.cat])

    
    def fit(self, X, y=None):

            
        if self.encode in ['ohe', 'ord']:        
            try:    
                self.name = X.name 
            except AttributeError:
                self.name = 'SibSp'
                
            dummy_df = pd.DataFrame({self.name: ['0']})
            self.enc.fit(dummy_df)
        
            
        return self

    
    @staticmethod
    def get_cat(thresh):
        return [str(x) for x in range(thresh+1)] + [f'>{thresh}']
        
    
    def transform(self, X):
        X = X.copy()
        X[X > self.thresh] = f'>{self.thresh}'
        
        X = pd.DataFrame(X).astype(str)
        
        if self.encode in ['ohe', 'ord']:
            X = self.enc.transform(X)
            
        return X
    
    def get_feature_names(self, input_features=None):
        if self.encode in ['ohe', 'ord']:
            return self.enc.get_feature_names(input_features)
        