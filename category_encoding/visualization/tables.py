import pandas as pd
import os
import glob
import time
from tqdm._tqdm import tqdm
from .utils import to_pandas

class Table:
    def __init__(self, results=None, path=None):
        if results is None:
            if path is None:
                raise ValueError
            results_list = glob.glob(os.path.join(path, '*.csv'))
            self.results = {os.path.split(i)[1][:-4]: pd.read_csv(i) for i in results_list}
        else:
            self.results = to_pandas(results)

    def save(self, path):
        ts = int(time.time())
        res_dir = 'res_{}'.format(ts)
        path = os.path.join(path, res_dir)
        os.makedirs(path)

        for name, df in self.results.items():
            df.to_csv(os.path.join(path, '{}.csv'.format(name)), index=False)
        
        return path

    def to_markdown(self):
        pass

    def to_latex(self):
        pass

        
