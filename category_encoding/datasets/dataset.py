import pandas as pd
class Dataset:
    def __init__(self, path, cat_cols, target_col):
        df = self.load(path)
        df = self.preprocessing(df)

        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        self.cat_cols = cat_cols
        self.feat_names = self.X.columns

    def load(self, path):
        return pd.read_csv(path)

    def preprocessing(self, df):
        return df