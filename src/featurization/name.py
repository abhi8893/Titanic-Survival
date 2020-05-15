from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from .helpers import has_pattern

class NameTitleExtractor(BaseEstimator, TransformerMixin):
    
    '''
    Extracts the titles from the name
    
    Parameters
    ----------
    
    # TODO: implement passing any excluding categories
    
    exclude_cat_type: str, default: 'type1', options: ['type1', 'type2'] 
        which type of categories to exclude from grouping to 'other'
    '''
    
    exclude_cat_opts = {
        'type1': ['Mr', 'Mrs', 'Miss', 'Master'],
        'type2': ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev']
    }
    
    def __init__(self, exclude_cat_type='type1', sparse=False, pattern=r'([A-Z][a-z]+)\.'):
        self.exclude_cat_type = exclude_cat_type
        self.exclude_cat = self.exclude_cat_opts[exclude_cat_type]
        self.cat = self.exclude_cat + ['other']
        self.pattern = pattern
        self.sparse = sparse
        self.ohe = OneHotEncoder([self.cat], drop=['other'], sparse=sparse)
        
    def fit(self, X, y=None):
                    
        try:    
            name = X.name 
        except AttributeError:
            name = 'Name'
            
        dummy_df = pd.DataFrame({name: ['Mr']})
        self.ohe.fit(dummy_df)
        
        return self
    
    def transform(self, X):
        X = X.apply(has_pattern, pattern=self.pattern, grp=1)

        X.replace(['Mlle', 'Ms'], 'Miss', inplace=True)
        X.replace(['Mme'], 'Mrs', inplace=True)
       
        X.loc[~X.isin(self.exclude_cat)] = 'other'
            
        return self.ohe.transform(pd.DataFrame(X))
    
    
    def get_feature_names(self, input_features):
        return self.ohe.get_feature_names(input_features)


            