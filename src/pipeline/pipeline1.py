from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ..preprocessing import NaNDropper
from .data_pipeline import DataPipeline
from sklearn.svm import SVC


prepare_data = Pipeline([
    ('nan_drpr', NaNDropper(['Embarked']))
])

clmn_trnsfrm = ColumnTransformer([
    ('clmn_drp', 'drop', ['Name', 'Ticket', 'Cabin']),
    ('enc', OneHotEncoder(drop='first'), ['Sex', 'Embarked']),
    ('imp', SimpleImputer(), ['Age'])
], 'passthrough')


preprocess_data = Pipeline([
    ('clmn_trnsfrm', clmn_trnsfrm),
    ('scale', StandardScaler())
])



mlmodel = SVC()

description = r'''
1. Drop Name, Ticket - requires Feature Engineering
2. OneHotEncoder for Sex, Embarked
3. Drop Cabin - requires Feature Engineering/(?And Not Imputation)
4. Age : Applied Mean Imputation and Mean Normalization
5. Drop NaN rows in Embarked
6. Applied StandardScaler at the end to all features

MLmodel: SVC'''


dp1 = DataPipeline(
    prepare_data, 
    preprocess_data, 
    mlmodel, 
    description=description,
    ycol='Survived')

dp1.name = 'dp1'
