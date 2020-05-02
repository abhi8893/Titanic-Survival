import numpy as np
import re
from .helpers import binarizer

def FE_SibSp(arr: np.array):
    arr = arr.copy()
    arr[arr>1] = 2
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    return arr


def FE_Parch(arr: np.array):
    arr = arr.copy()
    arr[arr>1] = 2
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    return arr

def FE_Ticket(x):
    x = x.str.replace(r'[^A-Za-z0-9\s]+', '')
    x = x.apply(lambda x: x.split(' ')[0] if not x.isdigit() else 'numeric')
    
    return x.values.reshape(-1, 1)

def FE_Name(x, pattern='([A-Z][a-z]+)\.'):
    x = x.apply(lambda x: re.search(pattern, x).group(1))
    x.replace(['Mlle', 'Ms'], 'Miss', inplace=True)
    x.replace(['Mme'], 'Mrs', inplace=True)
    x.loc[~x.isin(['Mr', 'Mrs', 'Miss', 'Master'])] = 'Special'
    return x.values.reshape(-1, 1)


def FE_Cabin(x):
    col1 = x.str[0].fillna('NC')
    return col1.values.reshape(-1, 1)

