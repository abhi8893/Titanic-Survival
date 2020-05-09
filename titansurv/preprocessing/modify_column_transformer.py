from sklearn.compose import ColumnTransformer
from sklearn.base import clone


def __convert_to_list(obj):
    if not isinstance(obj, list):
        return [obj]



def modify_transformer_cols(col_trnsfrmr: ColumnTransformer, append=False, inplace=False, **trnsfrmr_cols):
    if not inplace:
        new_col_trnsfrmr = clone(col_trnsfrmr)

    trnsfrmrs = new_col_trnsfrmr.transformers
    for i, [trnsfrmr_name, old_trnsfrmr, old_cols] in enumerate(trnsfrmrs):
        new_cols = trnsfrmr_cols.get(trnsfrmr_name, None)
        
        if new_cols is not None:
            if append:
                new_cols  = list(set().union(new_cols, old_cols))
        else:
            new_cols = old_cols
                            
        trnsfrmrs[i] = (trnsfrmr_name, old_trnsfrmr, new_cols)
        
    return new_col_trnsfrmr


def modify_transformer_est(col_trnsfrmr: ColumnTransformer, inplace=False, **trnsfrmr_estmtrs):
    if not inplace:
        new_col_trnsfrmr = clone(col_trnsfrmr)

    trnsfrmrs = new_col_trnsfrmr.transformers
    for i, [trnsfrmr_name, old_trnsfrmr, old_cols] in enumerate(trnsfrmrs):
        new_trnsfrmr = trnsfrmr_estmtrs.get(trnsfrmr_name, old_trnsfrmr)

        trnsfrmrs[i] = (trnsfrmr_name, new_trnsfrmr, old_cols)


    return new_col_trnsfrmr


def remove_transformer_by_name(col_trnsfrmr: ColumnTransformer, names: list, inplace=False):
    if not inplace:
        new_col_trnsfrmr = clone(col_trnsfrmr)

    trnsfrmrs = new_col_trnsfrmr.transformers
    for i, [trnsfrmr_name, old_trnsfrmr, old_cols] in enumerate(trnsfrmrs):
        if trnsfrmr_name in names:
            trnsfrmrs.pop(i)

    return new_col_trnsfrmr