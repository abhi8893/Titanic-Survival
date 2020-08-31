from pprint import PrettyPrinter 
import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn_pandas import DataFrameMapper

__pp = PrettyPrinter().pprint

def print_params(estimator):
    __pp(list(estimator.get_params()))


def group_low_count_cat(x, thresh, setval):

    '''Group categories with counts less than {thresh},
    and rename them to {setval}

    Parameters
    ----------

    thresh: int
        Minimum count of a low count category

    setval: object
        New value to set for the merged categories

    '''
    counts = x.value_counts()

    cat_grp = counts[counts <= thresh].index.tolist()

    counts = counts[::-1] 

    gt_thresh = counts.cumsum() >= thresh

    first_occur = gt_thresh.searchsorted(True)

    add_cat = gt_thresh.index[first_occur]

    cat_grp = cat_grp + [add_cat]

    def replace(v):
        if v in cat_grp:
            return setval
        else:
            return v

    return pd.DataFrame(x.apply(replace).value_counts())


def gen_grid(temp_grid, step=""):
    temp_grid = deepcopy(temp_grid) # Do I need this
    
    est_key = [k for k in temp_grid.keys() if k.startswith('__')][0].replace('__', '')
    if step == '':
        main_key = est_key
    else:
        main_key = f'{step}__{est_key}'
        
    res = {}
    res[main_key] = temp_grid.pop(f'__{est_key}')
    res.update({f'{main_key}__{k}': v for k, v in temp_grid.items()})
    
    return res


def param_from_temp_grid(temp_grid, return_dict=False, **kwargs):
    res = {k:gen_grid(v,**kwargs) for k, v in temp_grid.items()}
    
    if not return_dict:
        return list(res.values())

    return res

def get_dt_max_depth_vals(max_depth, nvals=5):
    return np.ceil((2**np.linspace(np.log2(4), np.log2(max_depth*1.5), nvals)))


def clmn_trnsfrmr_to_dfmapper(clmn_trnsfrmr, **kwargs):
    '''Converts ColumnTransformer instance to a DataFrameMapper instance
    
    '''
    dfmapper_input = []
    for (name, trnsfrmr, cols) in clmn_trnsfrmr.transformers:
        if trnsfrmr == 'drop':
            continue
        elif trnsfrmr == 'passthrough':
            trnsfrmr = None
            
        dfmapper_input.append((cols, trnsfrmr))
        
    remainder = clmn_trnsfrmr.remainder
        
    if remainder == 'passthrough':
        default = None
    elif remainder == 'drop':
        default = False
        
    return DataFrameMapper(dfmapper_input, default=default, **kwargs)


from ..config import get_config

config_params = get_config()

def load_data(subset='train', return_X_y=False):
    file = config_params[f'RAW_{subset.upper()}_DATA_FILE']
    df = pd.read_csv(file)
    if subset == 'train' and return_X_y:
        dfX = df.drop(['Survived'], axis=1)
        dfy = df.Survived

        return dfX, dfy 

    return df


from sklearn.model_selection import cross_val_score


def get_training_cv_score(pipe, dfX, dfy, **kwargs):
    print(f'Training score: {pipe.score(dfX, dfy)}')
    print(f'crossvalidation score: {cross_val_score(pipe, dfX, dfy, **kwargs).mean()}')

def get_best_param_score(search):
    print(f'Best param: {search.best_params_}')
    print(f'Best score: {search.best_score_}')










