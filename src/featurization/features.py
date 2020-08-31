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
    def __ticket_func(x):
        if not x.isdigit():
            return x.split(' ')[0]
        else:
            return 'numeric'
            
    x = x.apply(__ticket_func)
    
    return x.values.reshape(-1, 1)

# Name
# because can't use lambda
def has_pattern(x, pattern, grp):
    res = re.search(pattern, x)
    if grp is None:
        return res
    else:
        return res.group(grp)

def get_title(x, pattern=r'([A-Z][a-z]+)\.'):
    x = x.apply(has_pattern, pattern=pattern, grp=1)
    x.replace(['Mlle', 'Ms'], 'Miss', inplace=True)
    x.replace(['Mme'], 'Mrs', inplace=True)
    x.loc[~x.isin(['Mr', 'Mrs', 'Miss', 'Master'])] = 'Special'
    return x


# Cabin
def FE_Cabin(x):
    col1 = x.str[0].fillna('NC')
    return col1.values.reshape(-1, 1)


from .utils import add_columns
from sklearn.base import clone
familySize = clone(add_columns)

def get_feature_names(self):
    return 'FamilySize'

familySize.get_feature_names = get_feature_names
