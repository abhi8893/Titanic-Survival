import numpy as np
from sklearn.preprocessing import FunctionTransformer
import re


def is_numeric(x):
    if x == 'numeric':
        return 1
    else:
        return 0

def binary_enc(x):
    return np.array(list(map(is_numeric, x))).reshape(-1, 1)

binarizer = FunctionTransformer(binary_enc)


# because can't use lambda
def has_pattern(x, pattern, grp):
    res = re.search(pattern, x)
    if grp is None:
        return res
    else:
        return res.group(grp)