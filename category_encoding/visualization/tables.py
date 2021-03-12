import pandas as pd
from tqdm._tqdm_notebook import tqdm

def one_to_pandas(results):
    df = pd.DataFrame(columns=['data', 'transformer', 'model'] + list(results[0]['metric'].keys()))
    
    for res in tqdm(results):
        _df = pd.DataFrame(columns=df.columns)
        for metric_name, metric in res['metric'].items():
            _df[metric_name] = metric
        _df['data'] = res['data']
        _df['transformer'] = res['transformer']
        _df['model'] = res['model']
        df = df.append(_df)
    return df

def metrics_to_pandas(metrics):
    df = pd.DataFrame

    for metric in metrics:
        _df = pd.DataFrame()
        
