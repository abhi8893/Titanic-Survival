from .data_pipeline import DataPipeline
from ..featurization import NameTitleExtractor, CabinTypeExtractor, TicketTypeExtractor
from ..featurization import SibSpBinner, ParchBinner 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from ..preprocessing import NaNDropper


prepare_data = Pipeline([
    ('nan_drpr', NaNDropper(['Embarked']))
])


imp_enc = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder())
])


preprocess_noscale = ColumnTransformer([
    ('enc', OneHotEncoder(drop='first'), ['Sex']),
    ('imp_enc', imp_enc, ['Embarked']),
    ('imp', SimpleImputer(), ['Age', 'Fare']),
    ('pre_Name', NameTitleExtractor(), 'Name'),
    ('cabin_type', CabinTypeExtractor(), 'Cabin'),
    ('pre_Ticket', TicketTypeExtractor(), 'Ticket'),
    ('pre_SibSp', SibSpBinner(), ['SibSp']),
    ('pre_Parch', ParchBinner(), ['Parch']),
], 
    'passthrough')

preprocess_data = Pipeline([
    ('clmn_trnsfrm', preprocess_noscale),
    ('scale', StandardScaler())
])

mlmodel = SVC()

description = r'''
1. `Embarked`: Dropped NA rows and applied OneHotEncoding
2. `Age`: Mean imputation with SimpleImputer 
3. `Sex`: OneHotEncoding
4. `Name`: Categorised into ['Mr', 'Mrs', 'Miss', 'Master', 'Special']<br/> 
    5.1 Rename [Mlle, Ms] -> Miss      
    5.2 Rename [Mme] -> Mrs     
    5.3 Put the Rest -> other     
    Then performed OneHotEncoding
5. `Ticket`: categorized into [1: numeric, 0: else] <br/>
    6.1 Remove special characters but not space <br/>
    6.2 Replace numeric strings by 'numeric' <br/>
    6.3 Split on space and keep the first item <br/>
 Then applied binarizer for [1: numeric, 0: else]
6. `SibSp`: binned into [0, 1, >1] using SibSpBinner
7. `Parch`: binned into [0, 1, >1] using ParchBinner
8. Scaling all features at the last using StandardScaler

**MLmodel:** `SVC`
'''


dp2 = DataPipeline(
    prepare_data, 
    preprocess_data, 
    mlmodel, 
    description=description,
    ycol='Survived')

dp2.name = 'dp2'