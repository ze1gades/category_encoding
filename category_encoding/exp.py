import pandas as pd
import importlib
import inspect
import json
from tqdm._tqdm_notebook import tqdm
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

def init_transformer(encoder_config, cat_cols):
    
    if isinstance(encoder_config['encoder_cls'], str):
        module_str, cls_name = encoder_config['encoder_cls'].rsplit('.', 1)
        module = importlib.import_module(name=module_str)
        encoder_cls = getattr(module, cls_name)
    else:
        encoder_cls = encoder_config['encoder_cls']

    encoder = encoder_cls(cols=cat_cols, **encoder_config.get('fargs', {}))
    encoder_name = encoder.__class__.__name__

    if encoder_config.get('normalization', None) is None:
        return encoder_name, encoder
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

    return transformer_name, transformer

def init_model(model_config):
    
    if 'model' in model_config:
        model = model_config['model']
    elif inspect.isclass(model_config['model_cls']):
        model = model_config['model_cls'](**model_config.get('fargs', {}))
    else:
        raise ValueError('model_cls is not class!!!')

    model_name = model.__class__.__name__
    model_type = model_config['type']
    return model_name, model_type, model


def make_exp(data_configs, encoder_configs, model_configs):
    update_data_config(data_configs)

    exp_list = []
    for data in data_configs:
        for encoder in encoder_configs:
            transformer_name, transformer = init_transformer(encoder, data['cat_cols'])
            for model in model_configs:
                model_name, model_type, model = init_model(model)

                exp_list.append({
                    'X': data['X'],
                    'y': data['y'],
                    'cat_cols': data['cat_cols'],
                    'data': data['name'],
                    'transformer': transformer,
                    'transformer_name': transformer_name,
                    'model': model,
                    'model_name': model_name,
                    'model_type': model_type
                })
    
    return exp_list

def run_one_exp(exp, metric_configs):
    results = {
        'data': exp['data'],
        'model': exp['model_name'],
        'transformer': exp['transformer_name'],
        'metrics': dict()
    }

    for metric in metric_configs:
        metric_name = metric['metric_name']
        if metric_name == 'diff_metrics':
            metric_result = metrics.diff_metrics(
                model=exp['model'], 
                transformer=exp['transformer'], 
                X=exp['X'],
                y=exp['y'],
                cat_cols=exp['cat_cols'],
                **metric.get('fargs', {})
            )
            
        elif metric_name == 'shap':
            metric_result = metrics.get_shap_values(
                model=exp['model'], 
                transformer=exp['transformer'],
                X=exp['X'],
                y=exp['y']
            )
        elif metric_name == 'feature_importance':
            metric_result = metrics.get_feature_importance(
                model=exp['model'],
                transformer=exp['transformer'],
                X=exp['X'],
                y=exp['y'],
                type=exp['type']
            )
        else:
            raise ValueError

        results['metrics'][metric_name] = metric_result
    return results

def load_configs(config):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    return config

def run_exp(configs):
    configs = {i: load_configs(j) for i, j in configs.items()}
    experements = make_exp(
        data_configs=configs['datas'],
        encoder_configs=configs['encoders'], 
        model_configs=configs['models']
    )

    exp_results = []

    for exp in tqdm(experements):
        exp_results.append(run_one_exp(exp, metric_configs=configs['metrics']))

    return exp_results