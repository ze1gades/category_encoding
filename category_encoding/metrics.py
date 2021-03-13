import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

def get_feature_importances(model, feat_names, type='linear'):

    if type == 'linear':
        result = np.abs(model.coef_).mean(axis=0).tolist()
    elif type == 'tree':
        result = np.abs(model.feature_importances_)
    elif type == 'kernel':
        result = None
    else:
        raise ValueError('Wrong type!!!')

    return {
        'feature_importances': result, 
        'orig_feat_names': feat_names[0],
        'enc_feat_names': feat_names[1]
    }

def get_shap_values(model, transformer, data, feat_names, type='linear'):
    X = transformer.transform(data.X)
    if type == 'linear':
        explainer = shap.LinearExplainer(model, X)
        shap_values = np.abs(explainer.shap_values(X)).mean(axis=0)
    elif type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = np.abs(explainer.shap_values(X))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = shap_values.mean(axis=0)
    elif type == 'kernel':
        explainer = shap.KernelExplainer(model, X)
        shap_values = np.abs(explainer.shap_values(X)).mean(axis=0)
    else:
        raise ValueError('Wrong type!!!')

    return {
        'shap': shap_values,
        'orig_feat_names': feat_names[0],
        'enc_feat_names': feat_names[1]
    }

def diff_metrics(model, transformer, data, metric='roc_auc', **fargs):
    pipeline = Pipeline([
        ('transformer', clone(transformer)),
        ('model', clone(model))
    ])

    result = {
        'without_cat_cols': cross_val_score(model, data.X.drop(columns=data.cat_cols), data.y, scoring=metric, **fargs),
        'just_cat_cols': cross_val_score(pipeline, data.X[data.cat_cols], data.y, scoring=metric, **fargs),
        'with_cat_cols': cross_val_score(pipeline, data.X, data.y, scoring=metric, **fargs)
    }

    result['add_cat_improve'] = result['with_cat_cols'] / result['without_cat_cols'] - 1
    result['add_non_cat_improve'] = result['with_cat_cols'] / result['just_cat_cols'] - 1

    return result

