import kaggle
import os
import json
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from .dataset_configs import DEFAULT_DATASETS

def load(dataset_config, path='./data/'):
    if isinstance(dataset_config, str):
        dataset_config = DEFAULT_DATASETS[dataset_config]

    kaggle.api.dataset_download_files(dataset_config['dataset'], path, unzip=True)

    path = os.path.join(path, dataset_config['file_name'])

    df = pd.read_csv(path, **dataset_config.get('read_fargs', {}))

    if 'replace' in dataset_config:
        df = df.replace(dataset_config['replace'])

    df[dataset_config['str_cols']] = OrdinalEncoder(dtype=int).fit_transform(df[dataset_config['str_cols']])

    df.to_csv(path, index=False)

    return {
        'path': path,
        'cat_cols': dataset_config['cat_cols'],
        'target_col': dataset_config['target_col']
    }

def get_default_datasets_list():
    return DEFAULT_DATASETS.keys()