import pandas as pd
import numpy as np
import re


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

def chain(*iterables):
    res = []
    for i in iterables:
        if isinstance(i, (list, tuple)):
            res = res + list(i)
        else:
            res.append(i)
    return res

def to_latex(df, caption, maximize):
    df = df.round(4)
    for col in df.columns:
        extr = df[col].max() if maximize else df[col].min()
        df[col] = np.where(df[col] == extr, '\\textbf{{{:.4f}}}'.format(extr), df[col]).astype(str)
    s = df.to_latex(escape=False, caption=caption, column_format='|l|' + 'l|' * len(df.columns))
    s = re.sub(r'([_^])', r'\\\1', s)
    return s.replace("\\\n", "\\ \hline\n")