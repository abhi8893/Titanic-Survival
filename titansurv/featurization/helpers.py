import numpy as np
from sklearn.preprocessing import FunctionTransformer

@np.vectorize
def binary_enc(x):
    if x == 'numeric':
        return 1
    else:
        return 0

binarizer = FunctionTransformer(binary_enc)