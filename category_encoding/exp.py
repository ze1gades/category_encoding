import pandas as pd
import importlib
import inspect
import json
from joblib import Parallel, delayed
from tqdm._tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from category_encoding import metrics


def update_data_config(data_configs):

    for data in data_configs:
        if 'name' not in data:
            if 'path' not in data:
                raise ValueError('Wrong data config!!!')
        
        if ('X' not in data) and ('y' not in data):
            if 'df' not in data:
                if 'path' in data:
                    data['df'] = pd.read_csv(data['path'])
                else:
                    raise ValueError('Wrong data config!!!')
            data['X'] = data['df'].drop(columns=[data['target_col']])
            data['y'] = data['df'][data['target_col']]

def init_transformer(encoder_config, cat_cols, X, y):
    
    if isinstance(encoder_config['encoder_cls'], str):
        module_str, cls_name = encoder_config['encoder_cls'].rsplit('.', 1)
        module = importlib.import_module(name=module_str)
        encoder_cls = getattr(module, cls_name)
    else:
        encoder_cls = encoder_config['encoder_cls']

    encoder = encoder_cls(cols=cat_cols, **encoder_config.get('fargs', {}))
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
    
    transformer.fit(X, y)
    enc_feat_names = transformer.named_steps['encoder'].get_feature_names()
    
    mapping = {}
    for feat in transformer.named_steps['encoder'].mapping:
        for i in feat['mapping']:
            mapping[i] = feat['col']
    orig_feat_names = [mapping.get(i, i) for i in enc_feat_names]

    return transformer_name, transformer, [orig_feat_names, enc_feat_names]

def init_model(model_config, X, y):
    
    if 'model' in model_config:
        model = model_config['model']
    elif inspect.isclass(model_config['model_cls']):
        model = model_config['model_cls'](**model_config.get('fargs', {}))
    else:
        raise ValueError('model_cls is not class!!!')

    model_name = model.__class__.__name__
    model_type = model_config['type']
    model.fit(X, y)
    return model_name, model_type, model

def make_exp(data_configs, encoder_configs, model_configs):
    update_data_config(data_configs)

    exp_list = []
    for data in data_configs:
        for encoder in tqdm(encoder_configs):
            transformer_name, transformer, feat_names = init_transformer(encoder, data['cat_cols'], data['X'], data['y'])
            X_enc = transformer.transform(data['X'])
            for model in model_configs:
                model_name, model_type, model = init_model(model, X_enc, data['y'])

                exp_list.append({
                    'X': data['X'],
                    'y': data['y'],
                    'cat_cols': data['cat_cols'],
                    'data': data['name'],
                    'transformer': transformer,
                    'transformer_name': transformer_name,
                    'model': model,
                    'model_name': model_name,
                    'model_type': model_type,
                    'feat_names': feat_names
                })
    
    return exp_list

def run_one_exp(exp, metric_config):
    metric_result = {
        'data': exp['data'],
        'model': exp['model_name'],
        'transformer': exp['transformer_name'],
    }

    metric_name = metric_config['metric_name']
    if metric_name == 'diff_metrics':
        metric_result['metric'] = metrics.diff_metrics(
            model=exp['model'], 
            transformer=exp['transformer'], 
            X=exp['X'],
            y=exp['y'],
            cat_cols=exp['cat_cols'],
            **metric_config.get('fargs', {})
        )
        
    elif metric_name == 'shap':
        metric_result['metric'] = metrics.get_shap_values(
            model=exp['model'], 
            transformer=exp['transformer'],
            X=exp['X'],
            feat_names=exp['feat_names'],
            type=exp['model_type']
        )
    elif metric_name == 'feature_importances':
        metric_result['metric'] = metrics.get_feature_importances(
            model=exp['model'],
            feat_names=exp['feat_names'],
            type=exp['model_type']
        )
    else:
        raise ValueError

    return metric_result

def load_configs(config):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    return config

def run_exp(configs, n_jobs=-1):
    configs = {i: load_configs(j) for i, j in configs.items()}
    experements = make_exp(
        data_configs=configs['datas'],
        encoder_configs=configs['encoders'], 
        model_configs=configs['models']
    )

    exp_results = dict()

    for metric_config in configs['metrics']:
        print('Calculate metric with config:\n{}'.format(metric_config))
        exp_results[metric_config['metric_name']] = Parallel(n_jobs=n_jobs, verbose=10, prefer='threads')(
            delayed(run_one_exp)(exp, metric_config=metric_config) for exp in experements
        )
        # for exp in tqdm(experements):
        #     print(exp)
        #     exp_results[metric_config['metric_name']].append(run_one_exp(exp, metric_config=metric_config))

    return exp_results