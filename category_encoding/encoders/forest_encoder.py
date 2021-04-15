import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from category_encoding.encoders import TwoHotEncoder
from category_encoders import OneHotEncoder
from .utils import get_obj_cols

class RFEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        cols=None, 
        handle_missing='value', 
        handle_unknown='value',
        use_cat_names=False,
        return_df=True,
        max_subsets=None,
        max_depth=3,
        n_estimators=100,
        n_jobs=1
    ):
        self.cols = cols
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.use_cat_names = use_cat_names
        self.return_df = return_df
        self.max_subsets = max_subsets
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self._dim = X.shape[1]

        if self.cols is None:
            self.cols = get_obj_cols(X)

        self.dummy_encoder = OneHotEncoder(
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )

        self.dummy_encoder = self.dummy_encoder.fit(X)
        self.mapping = self.generate_mapping(X, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        return self

    def generate_mapping(self, X, y):
        X = self.dummy_encoder.transform(X.copy(deep=True))
        y = y.copy(deep=True)

        mapping = []

        for switch in self.dummy_encoder.mapping:
            col = switch.get('col')
            values = switch.get('mapping').copy(deep=True)
            
            if isinstance(self.max_depth, int):
                max_depth = self.max_depth
            elif isinstance(self.max_depth, float):
                max_depth = round(self.max_depth * values.shape[1])
            else:
                max_depth = min(self.max_depth[1], round(self.max_depth[0] * values.shape[1]))
            if max_depth == 0:
                continue

            forest = RandomForestClassifier(
                max_depth=max_depth, 
                n_estimators=self.n_estimators, 
                n_jobs=self.n_jobs
            )

            forest.fit(X[values.columns], y)

            subsets = self.get_subsets(forest.decision_path(values))
            subset_df = pd.DataFrame(
                data=subsets,
                index=values.index,
                columns=['{col}_subset_{i}'.format(col=col, i=i) for i in range(subsets.shape[1])]
            )

            base_df = values.join(subset_df)
            
            mapping.append({'col': col, 'mapping': base_df})

        return mapping

    def get_subsets(self, decision_path):
        subset_sizes = np.asarray(decision_path[0].sum(axis=0))[0]
        subsets = decision_path[0][:, subset_sizes != 1].toarray()

        subsets, freq = np.unique(subsets, return_counts=True, axis=1)
        subsets = subsets[:, np.argsort(-freq)]

        subset_sizes = subsets.sum(axis=0)
        subsets = subsets[:, np.argsort(subset_sizes)]

        if self.max_subsets is not None:
            subsets = subsets[:, :self.max_subsets]

        return subsets

    def transform(self, X, override_return_df=False):
        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim, ))

        if not list(self.cols):
            return X if self.return_df else X.values

        X = self.dummy_encoder.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.get_dummies(X)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def get_dummies(self, X_in):
        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col])
            base_df = base_df.set_index(X.index)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        X = X.reindex(columns=cols)

        return X

    def get_feature_names(self):
        if not isinstance(self.feature_names, list):
            raise ValueError(
                'Must transform data first. Affected feature names are not known before.')
        else:
            return self.feature_names