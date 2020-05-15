'''
TODO: Add argparse functionality
TODO: python tune_mlmodel.py --datapipe [name of datapipe] --mlmodel [list of names] --search_type <grid, rand>
'''
from src.tune import tune_mlmodel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
from src.pipeline import dp1
from src.pipeline import dp2
from src.config import get_config
from src.utils import load_data
import joblib
import os
import argparse
import csv
from datetime import datetime as dt


parser = argparse.ArgumentParser()
parser.add_argument('-dp','--datapipe', help='provide the name of the datapipeline')
parser.add_argument('-ml', '--mlmodels', nargs="+", 
                    help='provide the mlmodel(s) separated by space or provide "all" to tune all mlmodels')
parser.add_argument('-st', '--search_type', default='rand',
                    help='provide the type of search to perform', choices=['grid', 'rand'])





def main():
    config_params = get_config()
    hyperparams = joblib.load(config_params['HYPERPARAMS_MLMODEL_FILE'])

    available_pipelines= {
        'dp1': dp1,
        'dp2': dp2
    }

    args = parser.parse_args()

    datapipe = args.datapipe


    if (len(args.mlmodels) == 1) and (args.mlmodels[0] != 'all'):
        mlmodels = args.mlmodels
    else:
        mlmodels = 'all'

    mlmodels = args.mlmodels

    search_type = args.search_type


    dp = available_pipelines[datapipe]
    dfX, dfy = load_data('train', return_X_y=True)
    dfX, dfy = dp.prepare_data.fit_transform(dfX, dfy)
    search = tune_mlmodel(dp, dfX, dfy, hyperparams, mlmodels, search_type)
    search_dir = config_params['SEARCH_DIR']
    file_name = f'{search_type}_search_{datapipe}.pkl'
    search_file = os.path.join(search_dir, file_name)
    joblib.dump(search, search_file)

    tuning_logfile = config_params['TUNING_MLMODEL_LOGFILE']
    if os.path.exists(tuning_logfile):
        mode = 'a+'
    else:
        mode = 'w'


    log_dict = {'datapipe': datapipe, 'mlmodels': mlmodels, 'search_type': search_type}
    
    with open(tuning_logfile, mode) as f:
        writer = csv.DictWriter(f, 
            fieldnames=["datapipe", "mlmodels", "search_type"])
        if mode != 'a+':
            writer.writeheader()

        writer.writerow(log_dict)


if __name__ == '__main__':
    main()
