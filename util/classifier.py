"""
class for the construction and validation of classifier models
with a classmethod to initialize from a soundprocess object
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import cloudpickle
import plotly.express as px
import plotly.graph_objs as go
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


class ClassifierModel(object):
    def __init__(self, x, y, ts, kind, feature_names=None):
        """
        initialize a classifier model with exogenous variables X and target variable y
        x: array (n x p)
        y: array (n x 1)
        ts: array of timestamps
        kind: str, source of the x matrix
        feature_names: names for columns of X
        """
        self.classifiers = {}
        self.kind=kind
        self.x=None
        self.y=None
        self.ts=ts
        self.feature_names=feature_names
        self.setup(x, y)
    
    @property
    def feature_importance(self):
        pass

    @property
    def unique_labels(self):
        """
        return the unique label values in the Y vector
        """
        names = self.y.drop_duplicates().sort_values().values
        output = set(names).difference(set(('',)))
        return list(output)

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError

    @classmethod
    def from_fft(cls, soundprocess):
        X = soundprocess.zarray.T
        y = soundprocess.target.label
        ts = soundprocess.fft_times
        return cls(X, y, ts, 'fft', feature_names=soundprocess.stft.f)
    
    @classmethod
    def from_fftpc(cls, soundprocess):
        X = soundprocess.pc_scores
        y = soundprocess.target.label
        ts = soundprocess.fft_times
        return cls(X, y, ts, 'fftpc', feature_names=soundprocess.pc_names)

    @classmethod
    def from_fftpcsm(cls, soundprocess):
        X = soundprocess.pc_smoothed
        y = soundprocess.target.label
        ts = soundprocess.fft_times
        return cls(X, y, ts, 'fftpcsm', feature_names=soundprocess.pc_names)

    def setup(self, x, y):
        self.x = x
        self.y = pd.Series(y)
    
    def save_binary(self, filename):
        data = cloudpickle.dumps(self)
        with open(filename, 'wb') as f:
            f.write(data)

    @property
    def feature_importance(self):
        raise NotImplementedError
    
    def plot_variable_importance(self):
        imp = (
            self.feature_importance
            .reset_index()
            .rename(columns={'index': 'label'})
            .melt(id_vars=['label'])
        )
        fig = px.bar(imp, x='variable', y='value', color='label')
        fig.update_traces(width=1)
        return fig
    
    def predict(self, sp):
        """
        predict the class of the data
        x: soundprocess
        """
        match self.kind:
            case 'fft':
                x = sp.zarray.T
            case 'fftpc':
                x = sp.pc_scores
            case 'fftpcsm':
                x = sp.pc_smoothed
            case _:
                raise ValueError()
        predictions = {}
        for label, model in self.classifiers.items():
            assert isinstance(model, RandomForestClassifier)
            yhat = model.predict_proba(x)
            predictions.update({label: yhat[:,1]})
        pred = pd.DataFrame(predictions)
        return pred.assign(ts=self.ts)
    
    def plot_predictions(self, sp):
        pred = self.predict(sp)
        pred_long = pred.melt(id_vars=['ts'])
        fig = px.line(
            pred_long,x='ts', y='value', 
            color='variable', title='Predicted Labels',
            labels={'value': 'Label Probability', 'ts': 'Timestamp', 'variable': 'Label'}
        )
        return fig



class XGClassifier(ClassifierModel):
    def __init__(self, x, y, ts, kind, feature_names):
        super().__init__(x, y, ts, kind, feature_names)

    @property
    def X(self):
        return xgb.DMatrix(self.x)
    
    @property
    def Y(self):
        return xgb.DMatrix()

    def fit(self):
        """
        fit classifier for target y given X
        """
        for label in self.unique_labels:
            _m = xgb.Booster()
            self.classifiers.append(_m)


class RFClassifier(ClassifierModel):
    def __init__(self, x, y, ts, kind, feature_names):
        super().__init__(x, y, ts, kind, feature_names)

    def fit(self):
        """
        fit classifier for target y given X
        """
        for label in tqdm(self.unique_labels, desc='Generating binary chord classifiers'):
            _m = RandomForestClassifier(max_depth=4, n_estimators=1000)
            _m.fit(self.x, np.where(self.y == label, 1, 0))
            self.classifiers.update({label: _m})

    @property
    def feature_importance(self):
        imp = []
        labels = sorted(self.unique_labels)
        for label in labels:
            importances = self.classifiers[label].feature_importances_
            imp.append(importances)
        imp = pd.DataFrame(np.array(imp), index=labels, columns=self.feature_names)
        return imp
