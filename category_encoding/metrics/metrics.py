import numpy as np
import shap
from timeit import default_timer as timer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

class Metric:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data
    def compute(self):
        pass

class FeatureImportances(Metric):
    def __init__(self, model, data, model_type, **kwargs):
        super(FeatureImportances, self).__init__(model, data)

        if model_type not in ['linear', 'tree', 'kernel']:
            raise ValueError('Wrong model_type!!!')

        self.model_type = model_type
    
    def compute(self):
        if self.model_type == 'linear':
            feature_importances = np.abs(self.model.coef_).mean(axis=0).tolist()
        elif self.model_type == 'tree':
            feature_importances = np.abs(self.model.feature_importances_)
        elif self.model_type == 'kernel':
            feature_importances = None
        return {
            'feature_importance': feature_importances,
            'enc_feat_name': self.data.enc_feat_names,
            'orig_feat_name': self.data.orig_feat_names
        }

class ShapValues(Metric):
    def __init__(self, model, data, model_type, **kwargs):
        super(ShapValues, self).__init__(model, data)

        if model_type not in ['linear', 'tree', 'kernel']:
            raise ValueError('Wrong model_type!!!')

        self.model_type = model_type

    def compute(self):
        if self.model_type == 'linear':
            explainer = shap.LinearExplainer(self.model, self.data.X_enc)
            shap_values = np.abs(explainer.shap_values(self.data.X_enc)).mean(axis=0)
        elif self.model_type == 'tree':
            explainer = shap.TreeExplainer(self.model)
            shap_values = np.abs(explainer.shap_values(self.data.X_enc))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = shap_values.mean(axis=0)
        elif self.model_type == 'kernel':
            explainer = shap.KernelExplainer(self.model, self.data.X_enc)
            shap_values = np.abs(explainer.shap_values(self.data.X_enc)).mean(axis=0)
        
        return {
            'shap_value': shap_values,
            'enc_feat_name': self.data.enc_feat_names,
            'orig_feat_name': self.data.orig_feat_names
        }

class CrossVal(Metric):
    def __init__(self, model, data, transformer, cv_kwargs, **kwargs):
        super(CrossVal, self).__init__(model, data)
        self.transformer = transformer
        self.kwargs = cv_kwargs

    def compute(self):
        pipeline = Pipeline([
            ('transformer', clone(self.transformer)),
            ('model', clone(self.model))
        ])

        result = {
            'without_cat_cols': cross_val_score(self.model, self.data.X.drop(columns=self.data.cat_cols), self.data.y, **self.kwargs),
            'just_cat_cols': cross_val_score(pipeline, self.data.X[self.data.cat_cols], self.data.y, **self.kwargs),
            'with_cat_cols': cross_val_score(pipeline, self.data.X,  self.data.y, **self.kwargs)
        }

        result['add_cat_improve'] = result['with_cat_cols'] / result['without_cat_cols'] - 1
        result['add_non_cat_improve'] = result['with_cat_cols'] / result['just_cat_cols'] - 1

        return result

class ExecutionTime(Metric):
    def __init__(self, model, data, transformer, n_execution=1, **kwargs):
        super(ExecutionTime, self).__init__(model, data)
        self.transformer = transformer
        self.n_execution = n_execution

    def compute(self):
        result = {
            'transformer_fit_time': list(),
            'model_fit_time': list(),
            'transformer_apply_time': list(),
            'model_apply_time': list()
        }

        for _ in range(self.n_execution):
            step_result = self.get_one_step_timings()

            for time_name, time in step_result.items():
                result[time_name].append(time)

        return result

    def get_one_step_timings(self):
        result = dict()
        transformer = clone(self.transformer)
        model = clone(self.model)

        start_time = timer()
        transformer.fit(self.data.X, self.data.y)
        result['transformer_fit_time'] = timer() - start_time

        start_time = timer()
        X_enc = transformer.transform(self.data.X)
        result['transformer_apply_time'] = timer() - start_time

        start_time = timer()
        model.fit(X_enc, self.data.y)
        result['model_fit_time'] = timer() - start_time

        start_time = timer()
        model.predict_proba(X_enc)
        result['model_apply_time'] = timer() - start_time

        return result
