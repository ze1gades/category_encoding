from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import importlib
import pandas as pd
import numpy as np

def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)

def get_orig_feats(feat_names, enc_feat_names):
    orig_feat_names = []
    for i in enc_feat_names:
        if i in feat_names:
            orig_feat_names.append(i)
        else:
            orig_feat_names.append(i.rsplit('_')[0])
    return orig_feat_names

def init_transformer(encoder_config, data):
    
    if isinstance(encoder_config['encoder_cls'], str):
        module_str, cls_name = encoder_config['encoder_cls'].rsplit('.', 1)
        module = importlib.import_module(name=module_str)
        encoder_cls = getattr(module, cls_name)
    else:
        encoder_cls = encoder_config['encoder_cls']

    encoder = encoder_cls(cols=data.cat_cols, **encoder_config.get('fargs', {}))
    encoder_name = encoder.__class__.__name__

    if encoder_config.get('normalization', None) is None:
        transformer = Pipeline([
            ('encoder', encoder)
        ])
        transformer_name = encoder_name
    elif encoder_config['normalization'] == 'std':
        transformer = Pipeline([
            ('encoder', encoder),
            ('normalization', StandardScaler())
        ])
        transformer_name = '{} + StandardScaler'.format(encoder_name)
    elif encoder_config['normalization'] == 'min_max':
        transformer = Pipeline([
            ('encoder', encoder),
            ('normalization', MinMaxScaler())
        ])
        transformer_name = '{} + MinMaxScaler'.format(encoder_name)
    else:
        raise ValueError
    
    data.X_enc = transformer.fit_transform(data.X, data.y)

    data.enc_feat_names = transformer.named_steps['encoder'].get_feature_names()
    data.orig_feat_names = get_orig_feats(data.X.columns, data.enc_feat_names)

    return transformer_name, transformer

def entropy(x):
    return np.where(x == 0, 0, x * np.log(x))