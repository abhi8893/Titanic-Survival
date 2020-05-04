from .data_pipeline import DataPipeline
from ..featurization import FE_Cabin, FE_Name, FE_Parch, FE_SibSp, FE_Ticket
from ..featurization.helpers import binarizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ..preprocessing.transformers import NaNDropper


prepare_data = Pipeline([
    ('nan_drpr', NaNDropper(['Embarked']))
])

imp_scale = Pipeline([
    ('imp', SimpleImputer()),
    ('scaler', StandardScaler())
], 'passthrough')

# TODO: I should not hardcode these
pre_Name = Pipeline([
    ('featurize', FunctionTransformer(FE_Name)),
    ('enc', OneHotEncoder(categories=[['Mr', 'Mrs', 'Miss', 'Master', 'Special']],
                          drop='first'))
])

# TODO: Implement modify pipeline function for DRY
pre_Cabin = Pipeline([
    ('featurize', FunctionTransformer(FE_Cabin)),
    ('enc', OneHotEncoder(categories=[['A', 'B', 'C', 'D', 
                                      'E', 'F', 'G', 'T', 'NC']], 
                          drop='first'))
])

pre_Ticket = Pipeline([
    ('featurize', FunctionTransformer(FE_Ticket)),
    ('binarizer', binarizer)
])

pre_SibSp = Pipeline([
    ('binner', FunctionTransformer(FE_SibSp)),
    ('enc', OneHotEncoder(drop='first'))
])

pre_Parch = Pipeline([
    ('binner', FunctionTransformer(FE_Parch)),
    ('enc', OneHotEncoder(drop='first'))
])


preprocess_data = ColumnTransformer([
    ('enc', OneHotEncoder(drop='first'), ['Sex', 'Embarked']),
    ('imp_scaler', imp_scale, ['Age']),
    ('scaler', StandardScaler(), ['Fare']),
    ('pre_Name', pre_Name, 'Name'),
    ('pre_Cabin', pre_Cabin, 'Cabin'),
    ('pre_Ticket', pre_Ticket, 'Ticket'),
    ('Pre_SibSp', pre_SibSp, ['SibSp']),
    ('Pre_Parch', pre_Parch, ['Parch'])
], 
    'passthrough')

mlmodel = RandomForestClassifier()

description = r'''The following were the preprocessing steps used: 
1. **Embarked**: Dropped NA rows and applied OneHotEncoding
2. **Age** : Applied Mean Imputation and Mean Normalization
3. **Fare**: Mean Normalization
4. **Sex**: OneHotEncoding
5. **Name**: Categorised into ['Mr', 'Mrs', 'Miss', 'Master', 'Special']<br/> 
    5.1 Rename [Mlle, Ms] -> Miss      
    5.2 Rename [Mme] -> Mrs     
    5.3 Put the Rest -> Special     
    Then performed OneHotEncoding
6. **Ticket** categorized into [1: numeric, 0: else] <br/>
    6.1 Remove special characters but not space <br/>
    6.2 Replace numeric strings by 'numeric' <br/>
    6.3 Split on space and keep the first item <br/>
 Then applied binarizer for [1: numeric, 0: else]
7. **SibSp** binned into [0, 1, >1] and applied OneHotEncoding
8. **Parch** binned into [0, 1, >1] and applied OneHotEncoding

Tuned ML model: **RandomForestClassifier** using GridSearchCV'''



pipeline1 = DataPipeline(
    prepare_data, 
    preprocess_data, 
    mlmodel, 
    description=description,
    ycol='Survived')