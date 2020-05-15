from src.config import get_config
from src.utils import load_data
import pandas as pd
from datetime import datetime 
import os

config_params = get_config()

def get_submission_name(dp, add_time=False):
    if not add_time:
        return '_'.join([dp.name, "submission.csv"])
    else:
        now = datetime.now()
        now = now.strftime("%d%m%y%H%M")
        return  '_'.join([dp.name, now, "submission.csv"])
        
        

def create_submission(pred, outname=None, outdir=None):
    ''' 
    Create submission dataframe and optionally write out a file
    
    '''
    psngr_id = load_data('test')['PassengerId']
    sub_df = pd.DataFrame({'PassengerId': psngr_id, 'Survived': pred})
    
    if outname is not None:
        if outdir is None:
            submission_dir = config_params['SUBMISSION_DIR']
        else:
            submission_dir = outdir
            
    elif outdir is not None:
        raise ValueError('Please provide outname')
            
    outfile = os.path.join(submission_dir, outname)    
    sub_df.to_csv(outfile, index=False)
    print(f'Created submission file at {outfile}')
        
    return sub_df