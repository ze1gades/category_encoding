import pandas as pd
import os
import time
from tqdm._tqdm import tqdm

def to_pandas(results):
    df = dict()
    for res_name, result in results.items():
        df[res_name] = pd.DataFrame(columns=['dataset', 'transformer', 'model'] + list(result[0]['metric'].keys()))
        
        for res in tqdm(result):
            _df = pd.DataFrame(columns=df[res_name].columns)
            for metric_name, metric in res['metric'].items():
                _df[metric_name] = metric
            _df['dataset'] = res['dataset']
            _df['transformer'] = res['transformer']
            _df['model'] = res['model']
            df[res_name] = df[res_name].append(_df)
    return df

def save(df_dict, path):
    ts = int(time.time())
    res_dir = 'res_{}'.format(ts)
    path = os.path.join(path, res_dir)
    os.makedirs(path)

    for name, df in df_dict.items():
        df.to_csv(os.path.join(path, '{}.csv'.format(name)), index=False)
    
    return path

        
