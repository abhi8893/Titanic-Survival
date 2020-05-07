import numpy as np
from sklearn.preprocessing import FunctionTransformer


def is_numeric(x):
    if x == 'numeric':
        return 1
    else:
        return 0

def binary_enc(x):
    return np.array(list(map(is_numeric, x))).reshape(-1, 1)

binarizer = FunctionTransformer(binary_enc)