
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def __adder(df):
    return pd.DataFrame(df.sum(axis=1))

add_columns = FunctionTransformer(__adder)

