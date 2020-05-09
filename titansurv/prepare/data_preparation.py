from ..preprocessing import NaNDropper
from sklearn.pipeline import Pipeline

drop_NA_Embarked = Pipeline([
    ('nan_drpr', NaNDropper(['Embarked']))
])
