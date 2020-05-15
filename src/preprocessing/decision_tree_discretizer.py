from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class DecisionTreeDiscretizer(DecisionTreeClassifier):
    
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort='deprecated',
                 ccp_alpha=0.0,
                 encoder='label'):
        super().__init__(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        class_weight=class_weight,
        random_state=random_state,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        presort=presort,
        ccp_alpha=ccp_alpha)
            
        self.encoder = encoder

        
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        super().fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)

        prob = self.predict_proba(X)
        if self.encoder == 'label':
            self.enc = LabelEncoder().fit(prob)
        elif self.encoder == 'ohe':
            self.enc = OneHotEncoder(sparse=False).fit(prob)

        return self


    def transform(self, X, y=None):
        prob = self.predict_proba(X)
        return self.enc.transform(prob)
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)



