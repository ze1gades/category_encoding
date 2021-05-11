import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from .utils import get_obj_cols, entropy


class InformativeEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        cols=None, 
        handle_missing='value', 
        handle_unknown='value',
        return_df=True,
        max_subset_size=None,
        max_subsets=None, 
        criterion='gini'
    ):
        self.cols = cols
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.return_df = return_df
        self.max_subset_size = max_subset_size
        self.max_subsets = max_subsets
        self.criterion = criterion

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
        self.mapping = self.generate_mapping(X, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        return self

    def generate_mapping(self, X=None, y=None):
        mapping = []

        if self.max_subsets is not None:
            X = self.ordinal_encoder.transform(X)

        for switch in self.ordinal_encoder.mapping:
            col = switch.get('col')
            values = switch.get('mapping').copy(deep=True)

            if self.handle_missing == 'value':
                values = values[values > 0]
            
            if len(values) == 0:
                continue

            index = values.values

            base_matrix = np.eye(len(values), dtype=np.int)
            base_matrix = self.get_subsets(base_matrix)
            new_columns = ['{col}_{i}'.format(col=col, i=i) for i in range(base_matrix.shape[1])]

            base_df = pd.DataFrame(data=base_matrix, columns=new_columns, index=index)

            if self.max_subsets is not None:
                y_sum = y.groupby(X[col]).sum().reindex(index).values
                y_count = X.groupby(col).size().reindex(index).values
                info = self.cut_info(
                    base_df.values,
                    y_sum,
                    y_count
                )

                if isinstance(self.max_subsets, float):
                    max_subsets = round(self.max_subsets * base_df.shape[0])
                else:
                    max_subsets = self.max_subsets

                base_matrix = base_matrix[:, np.argsort(info)][:, :max_subsets]

                new_columns = ['{col}_{i}'.format(col=col, i=i) for i in range(base_matrix.shape[1])]
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

    def get_subsets(self, base_matrix):
        max_subset_size = self.max_subset_size or base_matrix.shape[1]

        subsets = base_matrix

        for _ in range(1, max_subset_size):
            subsets = base_matrix[:, :, np.newaxis] + subsets[:, np.newaxis, :]
            subsets = np.unique(subsets.clip(0, 1).reshape(base_matrix.shape[0], -1), axis=1)

        return subsets

    def cut_info(self, subsets, y_sum, y_count):
        sum_l = np.sum(subsets * y_sum[:, np.newaxis], axis=0)
        count_l = np.sum(subsets * y_count[:, np.newaxis], axis=0)
        p_l = sum_l / count_l

        sum_r = np.sum((1 - subsets) * y_sum[:, np.newaxis], axis=0)
        count_r = np.sum((1 - subsets) * y_count[:, np.newaxis], axis=0)
        p_r = sum_r / count_r

        crit_l, crit_r = self.compute_criterion(p_l, p_r)

        return count_l * crit_l + count_r * crit_r

    def compute_criterion(self, p_l, p_r):
        if self.criterion == 'gini':
            crit_l = 1. - p_l**2 - (1 - p_l)**2
            crit_r = 1. - p_r**2 - (1 - p_r)**2
        elif self.criterion == 'entropy':
            crit_l = -(entropy(p_l) + entropy(1 - p_l))
            crit_r = -(entropy(p_r) + entropy(1 - p_r))
        else:
            raise ValueError
        
        return crit_l, crit_r

        
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
