import os
import category_encoding.datasets as datasets
from sklearn.pipeline import Pipeline
import inspect

def get_datasets(dataset_configs):
    data = dict()
    for config in dataset_configs:
        if isinstance(config, str):
            data[config] = getattr(datasets, config)('./')
        else:
            name = config.get('name', os.path.split(config['path'])[1])
            config.pop('name', None)
            data[name] = datasets.Dataset(**config)
    return data

def init_model(model_config, data):
    
    if 'model' in model_config:
        model = model_config['model']
    elif inspect.isclass(model_config['model_cls']):
        model = model_config['model_cls'](**model_config.get('fargs', {}))
    else:
        raise ValueError('model_cls is not class!!!')

    model_name = model.__class__.__name__
    model_type = model_config['type']

    model.fit(data.X_enc, data.y)
    
    return model_name, model_type, model