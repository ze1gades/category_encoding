from .dataset import Dataset
from sklearn.preprocessing import OrdinalEncoder
from .default_configs import DEFAULT_KAGGLE_CONFIGS as CONFIGS
import kaggle
import zipfile
import pandas as pd
import os
import shutil
import time

class KaggleDataset(Dataset):
    def __init__(self, path):
        cat_cols = CONFIGS[self.__class__.__name__]['cat_cols']
        target_col = CONFIGS[self.__class__.__name__]['target_col']

        download_path = self.make_download_dir(path)
        path = self.download(download_path)
        super(KaggleDataset, self).__init__(path, cat_cols, target_col)
        self.clear(download_path)
    
    def download(self, path):
        return path

    def make_download_dir(self, path):
        ts = int(time.time())
        tmp_dir = 'tmp_{}'.format(ts)
        download_path = os.path.join(path, tmp_dir)
        os.makedirs(download_path)
        return download_path

    def clear(self, path):
        shutil.rmtree(path)

class TelcoDataset(KaggleDataset):
    def __init__(self, path):
        super(TelcoDataset, self).__init__(path)

    def download(self, path):
        kaggle.api.dataset_download_files('blastchar/telco-customer-churn', path, unzip=True)
        return os.path.join(path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    def load(self, path):
        return pd.read_csv(path, index_col=0)

    def preprocessing(self, df):
        df = df.replace({'TotalCharges': {' ': '0'}})

        str_cols = CONFIGS[self.__class__.__name__]['str_cols']
        df[str_cols] = OrdinalEncoder(dtype=int).fit_transform(df[str_cols])

        return df

class AdultDataset(KaggleDataset):
    def __init__(self, path):
        super(AdultDataset, self).__init__(path)

    def download(self, path):
        kaggle.api.dataset_download_files('wenruliu/adult-income-dataset', path, unzip=True)
        return os.path.join(path, 'adult.csv')

    def preprocessing(self, df):

        str_cols = CONFIGS[self.__class__.__name__]['str_cols']
        df[str_cols] = OrdinalEncoder(dtype=int).fit_transform(df[str_cols])

        return df

class EmployeeDataset(KaggleDataset):
    def __init__(self, path):
        super(EmployeeDataset, self).__init__(path)

    def download(self, path):
        kaggle.api.competition_download_files('amazon-employee-access-challenge', path)
        with zipfile.ZipFile(os.path.join(path, 'amazon-employee-access-challenge.zip'), 'r') as z:
            z.extract('train.csv', path)
        return os.path.join(path, 'train.csv')

class KickedDataset(KaggleDataset):
    def __init__(self, path):
        super(KickedDataset, self).__init__(path)
    
    def download(self, path):
        kaggle.api.competition_download_files('DontGetKicked', path)
        with zipfile.ZipFile(os.path.join(path, 'DontGetKicked.zip'), 'r') as z:
            z.extract('training.csv', path)
        return os.path.join(path, 'training.csv')

    def load(self, path):
        return pd.read_csv(path, index_col=[0, 2])

    def preprocessing(self, df):
        str_cols = CONFIGS[self.__class__.__name__]['str_cols']
        df[str_cols] = df[str_cols].fillna('NoData')
        df = df.fillna(-1)
        df[str_cols] = OrdinalEncoder(dtype=int).fit_transform(df[str_cols])

        return df

class CreditDataset(KaggleDataset):
    def __init__(self, path):
        super(CreditDataset, self).__init__(path)
    
    def download(self, path):
        kaggle.api.competition_download_file('home-credit-default-risk', 'application_train.csv', path, quiet=True)
        with zipfile.ZipFile(os.path.join(path, 'application_train.csv.zip'), 'r') as z:
            z.extract('application_train.csv', path)
        return os.path.join(path, 'application_train.csv')

    def load(self, path):
        return pd.read_csv(path, index_col=0)

    def preprocessing(self, df):
        str_cols = CONFIGS[self.__class__.__name__]['str_cols']
        df[str_cols] = OrdinalEncoder(dtype=int).fit_transform(df[str_cols].fillna('NoData'))
        return df.fillna(-1)