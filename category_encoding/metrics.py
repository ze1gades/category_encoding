import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def get_feature_importance(model, transformer, X, y, type='linear'):
    X = transformer.fit_transform(X, y)
    model.fit(X, y)
    features = X.columns

    if type == 'linear':
        return {i: j for i, j in zip(features, np.abs(model.coef_).mean(axis=0).tolist())}
    elif type == 'tree':
        return {i: j for i, j in zip(features, model.feature_importances_)}
    else:
        raise ValueError('Wrong type!!!')

def get_shap_values(model, transformer, X, y):
    print(model)
    X = transformer.fit_transform(X, y)
    model.fit(X, y)
    explainer = shap.Explainer(model)
    return explainer(X)

def diff_metrics(model, transformer, X, y, cat_cols=[], metric='roc_auc', **fargs):
    pipeline = Pipeline([
        ('transformer', transformer),
        ('model', model)
    ])

    result = {
        'without_cat_cols': cross_val_score(model, X.drop(columns=cat_cols), y, scoring=metric, **fargs),
        'just_cat_cols': cross_val_score(pipeline, X[cat_cols], y, scoring=metric, **fargs),
        'with_cat_cols': cross_val_score(pipeline, X, y, scoring=metric, **fargs)
    }

    result['add_cat_improve'] = result['with_cat_cols'] / result['without_cat_cols'] - 1
    result['add_non_cat_improve'] = result['with_cat_cols'] / result['just_cat_cols'] - 1

    return result

