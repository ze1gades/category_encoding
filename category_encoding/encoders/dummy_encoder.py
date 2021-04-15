import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from .utils import get_obj_cols


class TwoHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        cols=None, 
        handle_missing='value', 
        handle_unknown='value',
        use_cat_names=False,
        return_df=True
    ):
        self.cols = cols
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.use_cat_names = use_cat_names
        self.return_df = return_df

    def fit(self, X, y=None, **kwargs):
        self._dim = X.shape[1]

        if self.cols is None:
            self.cols = get_obj_cols(X)

        self.ordinal_encoder = OrdinalEncoder(
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )

        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        self.mapping = self.generate_mapping()

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        return self

    def generate_mapping(self):
        mapping = []

        for switch in self.ordinal_encoder.mapping:
            col = switch.get('col')
            values = switch.get('mapping').copy(deep=True)

            if self.handle_missing == 'value':
                values = values[values > 0]
            
            if len(values) == 0:
                continue

            index = []
            new_columns = []

            n_col_name = str(col)
            for i, (cat_name, class_) in enumerate(values.iteritems()):
                if self.use_cat_names:
                    n_col_name = '_'.join([n_col_name, str(cat_name)])
                else:
                    n_col_name = '_'.join([n_col_name, str(class_)])

                if i % 2:
                    new_columns.append(n_col_name)
                    n_col_name = str(col)
                
                index.append(class_)
            
            if n_col_name != str(col):
                new_columns.append(n_col_name)

            index = index[::2] + index[1::2]

            base_matrix = np.eye(len(new_columns), dtype=np.int)
            base_matrix = np.vstack((base_matrix, -base_matrix))[:len(index), :]

            base_df = pd.DataFrame(data=base_matrix, columns=new_columns, index=index)

            if self.handle_unknown == 'value':
                base_df.loc[-1] = 0
            elif self.handle_unknown == 'return_nan':
                base_df.loc[-1] = np.nan

            if self.handle_missing == 'return_nan':
                base_df.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                base_df.loc[-2] = 0

            mapping.append({'col': col, 'mapping': base_df})
        return mapping
                
                
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

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.get_dummies(X)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def get_feature_names(self):
        if not isinstance(self.feature_names, list):
            raise ValueError(
                'Must transform data first. Affected feature names are not known before.')
        else:
            return self.feature_names

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
