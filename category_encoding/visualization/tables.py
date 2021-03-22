import pandas as pd
import os
import glob
import time
from tqdm._tqdm import tqdm
from .utils import to_pandas, to_latex, chain

class Table:
    def __init__(self, results=None, path=None):
        if results is None:
            if path is None:
                raise ValueError
            results_list = glob.glob(os.path.join(path, '*.csv'))
            self.results = {os.path.split(i)[1][:-4]: pd.read_csv(i) for i in results_list}
        else:
            self.results = to_pandas(results)

    def __getitem__(self, key):
        return self.results[key]

    def append(self, b):
        for k, v in b.results.items():
            if k in self.results:
                self.results[k] = self.results[k].append(v)
            else:
                self.results[k] = v

    def save(self, path):
        ts = int(time.time())
        res_dir = 'res_{}'.format(ts)
        path = os.path.join(path, res_dir)
        os.makedirs(path)

        for name, df in self.results.items():
            df.to_csv(os.path.join(path, '{}.csv'.format(name)), index=False)
        
        return path

    def get_metrics_list(self):
        return list(self.results.keys())

    def format(self, metric, key_col, columns, index, values, maximize, caption_form):
        df = self.results[metric]
        select_cols = chain(key_col, columns, index, values)
        group_cols = chain(key_col, columns, index)
        df = df[select_cols].groupby(group_cols).mean().reset_index()
        
        keys = df[key_col].unique().tolist()

        if isinstance(values, str):
            values = [values]

        if isinstance(maximize, bool):
            maximize = {v: maximize for v in values}
        elif isinstance(maximize, (list, tuple)):
            maximize = dict(zip(values, maximize))

        s = ""
        for val in values:
            for k in keys:
                t = df[df[key_col] == k].pivot(index=index, columns=columns, values=val)
                caption = caption_form.format(key=k, value=val)
                s = s + to_latex(t, caption, maximize[val])
        return s

    def to_markdown(self):
        pass

    def to_latex(self, t, caption):
        pass

        
