import pandas as pd
import os
import time
from tqdm._tqdm import tqdm

def process_metrics(metrics):
    data_frames = dict()

    for metric in metrics:
        df = pd.DataFrame(metric['result'])
        df['model'] = metric['model_name']

        metric_name = metric['metric_name']
        if metric_name in data_frames:
            data_frames[metric_name] = data_frames[metric_name].append(df)
        else:
            data_frames[metric_name] = df
    return data_frames

def to_pandas(results):
    data_frames = dict()
    
    for res in results:
        dfs = process_metrics(res['metrics'])
        for metric, df in dfs.items():
            df['dataset'] = res['dataset']
            df['transformer'] = res['transformer']
            if metric in data_frames:
                data_frames[metric] = data_frames[metric].append(df)
            else:
                data_frames[metric] = df 

    return data_frames

def save(df_dict, path):
    ts = int(time.time())
    res_dir = 'res_{}'.format(ts)
    path = os.path.join(path, res_dir)
    os.makedirs(path)

    for name, df in df_dict.items():
        df.to_csv(os.path.join(path, '{}.csv'.format(name)), index=False)
    
    return path

        
