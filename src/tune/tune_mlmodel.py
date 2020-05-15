from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import pandas as pd
from ..pipeline import dp1
from ..pipeline import dp2
from ..config import get_config
from ..utils import load_data
import joblib
import os

config_params = get_config()
HYPERPARAMS_MLMODEL_FILE = config_params['HYPERPARAMS_MLMODEL_FILE']


#def get_available_datapipes

available_pipelines= {
    'dp1': dp1,
    'dp2': dp2
}


def load_hyperparams_mlmodel():
    return joblib.load(HYPERPARAMS_MLMODEL_FILE)

def tune_mlmodel(dp, dfX, dfy, hyperparams,
                 mlmodels='all', search_type='rand', **kwargs):
    '''
    Tune the machine learning model in a pipeline
    
    Parameters
    ----------
    dp: DataPipeline
        
    dfX: pd.DataFrame
    dfy: pd.DataFrame
        
    hyperparams: dict or list of dicts, default: 'autoload'
        'autoload' will load the param_dict from the HYPERPARAMS_MLMODEL_FILE
        else the supply a list of dicts
        
    mlmodels: list default: 'all'
        provide a list of mlmodel names to tune
        'all' will tune all mlmodels
        
    search: str default: 'rand'
        the type of search to perform i.e. 'rand' -> RandomizedSearchCV, 'grid' -> GridSearchCV
    '''
        
    if mlmodels != 'all':
        if type(hyperparams) is dict:
            param_grid = [v for k, v in hyperparams.items() if k in mlmodels]
        else:
            param_grid = hyperparams
    
    pipe = dp.get_pipeline()

    if search_type == 'grid':
        search = GridSearchCV(pipe, param_grid, **kwargs)
    elif search_type == 'rand':
        search = RandomizedSearchCV(pipe, param_grid, **kwargs)
    
    return search.fit(dfX, dfy)