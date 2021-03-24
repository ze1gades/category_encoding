from .utils import get_datasets, init_model
from category_encoding.encoders import init_transformer
import category_encoding.metrics as metrics
from joblib import Parallel, delayed
from tqdm._tqdm import tqdm

def make_exp(data, transformer, configs):
    exps = []
    for model_config in configs['model']:
        model_name, model_type, model = init_model(model_config, data)
        for metric_name in configs['metric']:
            metric_fargs = {}
            if not isinstance(metric_name, str):
                metric_name, metric_fargs = metric_name

            metric_cls =  getattr(metrics, metric_name)
            metric = metric_cls(
                data=data,
                model=model,
                transformer=transformer,
                model_type=model_type,
                **metric_fargs
            )
            exps.append({
                'model_name': model_name,
                'metric_name': metric_name,
                'metric': metric
            })
    return exps

def run_one_exp(exp):
    exp['result'] = exp['metric'].compute()
    return exp

def run_exp(configs, n_jobs=1):
    datasets = get_datasets(configs['data'])
    exp_results = []
    for dataset, data in datasets.items():
        for encoder_config in tqdm(configs['encoder'], 'Process: {}'.format(dataset)):
            transformer_name, transformer = init_transformer(encoder_config, data)
            exps = make_exp(data, transformer, configs)

            exp_results.append({
                'dataset': dataset,
                'transformer': transformer_name,
                'metrics': Parallel(n_jobs=n_jobs, prefer='threads')(
                    delayed(run_one_exp)(exp) for exp in exps
                )
            })
    return exp_results
