from sklearn.pipeline import Pipeline
from sklearn.base import clone
from IPython.display import display, Markdown
import warnings
import pandas as pd
import re


class DataNotPreparedError(Exception):
    pass

class DataNotPreprocessedError(Exception):
    pass

class DataPipelineNotFittedError(Exception):
    pass

class DataPipeline(Pipeline):

    def __init__(self, prepare_data, preprocess_data, mlmodel, data=None, ycol=None, description=None):

        # BUG: Cloning the estimators either prepare_data, preprocess_data, mlmodel
        #      raises RunTimeError while cloning DataPipeline object
        self.__prepare_data = prepare_data
        self._is_prepared = False
        self.__preprocess_data = preprocess_data
        self._is_preprocessed = False
        self.__mlmodel = mlmodel
        self.__data = data
        self.ycol = ycol
        self.description = description
        

        self.__transformers = [
            ('preprocessing', preprocess_data),
            ('train', mlmodel)
        ]

        super().__init__(self.__transformers)
        self._is_fitted = False

        if data is None:
            warnings.warn('Please set the data first using set_data method!')
    


    def __repr__(self):
        return 'DataPipeline'

    def get_description(self, markdown=False):
        if markdown:
            display(Markdown(self.description))
        else:
            print(self.description)


    def _get_X_y(self):
        X = self.data.drop(self.ycol, axis=1)
        y = self.data.loc[:, self.ycol].copy()

        return X, y


    @property
    def data(self):
        return self.__data


    # TODO: Make a general way of setting any attribute
    @data.setter
    def data(self, data):
        self.__init__(
            self.prepare_data,
            self.preprocess_data,
            self.mlmodel,
            data,
            self.ycol,
            self.description
        )

    
    def set_data(self, data):
        self.data = data

    @property
    def prepare_data(self):
        return self.__prepare_data

    @prepare_data.setter
    def prepare_data(self, prepare_data):
        self.__init__(
            prepare_data,
            self.preprocess_data,
            self.mlmodel,
            self.data,
            self.ycol,
            self.description
        )


    @property
    def preprocess_data(self):
        return self.__preprocess_data

    @preprocess_data.setter
    def preprocess_data(self, preprocess_data):
        self.__init__(
            self.prepare_data,
            preprocess_data,
            self.mlmodel,
            self.data,
            self.ycol,
            self.description
        )


    @property
    def mlmodel(self):
        return self.__mlmodel

    @mlmodel.setter
    def mlmodel(self, mlmodel):
        self.__init__(
            self.prepare_data,
            self.preprocess_data,
            mlmodel,
            self.data,
            self.ycol,
            self.description
        )


    def prepare(self, *args, **kwargs):
        if self._is_prepared:
            return self.X, self.y

        self.X, self.y = self.prepare_data.fit_transform(*self._get_X_y(), *args, **kwargs)
        self._is_prepared = True
        return self.X, self.y

    def preprocess(self, *args, **kwargs):
        if not self._is_prepared:
            raise DataNotPreparedError('Please prepare the data first using the prepare method!')

        if self._is_preprocessed:
            return self.preprocessed
        
        self.preprocessed = self.preprocess_data.fit_transform(self.X, self.y, *args, **kwargs)
        self._is_preprocessed = True

        return self.preprocessed

    def train_mlmodel(self, *args, **kwargs):
        if not self._is_prepared:
            raise DataNotPreprocessedError('Please preprocess the data first using the preprocess method!')

        return self.mlmodel.fit(self.preprocessed, self.y)

    def get_pipeline(self):

        return clone(Pipeline(self.__transformers))


    def _check_dp_is_prepared(self):
        if not self._is_prepared:
            raise DataNotPreparedError('Please prepare the data first using the prepare method!')

    def _check_dp_is_fitted(self):
        if not self._is_fitted:
            raise DataPipelineNotFittedError('Please fit the data first using the fit method!')


    def fit(self, X=None, y=None, *args, **kwargs):


        if X is None and y is None:
            self._check_dp_is_prepared()
            fit = super().fit(self.X, self.y, *args, **kwargs)
    
        else:
            fit = super().fit(X, y, *args, **kwargs)
        
        self._is_fitted = True
        self.train_score = None

        return fit
        

    def score(self, X=None, y=None, *args, **kwargs):
        if X is None and y is None:
            self._check_dp_is_fitted()
            if self.train_score is None:
                self.train_score = super().score(self.X, self.y, *args, **kwargs)
                return self.train_score
        else:
            return super().score(X, y, *args, **kwargs)


    @staticmethod
    def __remove_duplicate_params(param_dict):
        keys = list(param_dict.keys())
        for k in keys:
            if re.search(r'^((\bmlmodel_)|(\bpreprocess_data_))(.*)', k):
                param_dict.pop(k)

        return param_dict


    def get_params(self, *args, **kwargs):
        out = super().get_params(*args, **kwargs)
        return self.__remove_duplicate_params(out)
    
        


    