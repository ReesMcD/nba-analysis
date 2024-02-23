from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb
import numpy as np
import shap


class GradientBoostModeler:
    def __init__(self, X, y, SEED):
        self.SEED = SEED
        self.model = None
        self.predictions = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=SEED)
        self.dtrain = xgb.DMatrix(self.X_train, self.y_train, enable_categorical=True)
        self.dtest = xgb.DMatrix(self.X_test, self.y_test, enable_categorical=True)

    def train(self, space, early_stopping_rounds=None):
        n = int(space['num_boost_round'])

        if 'max_depth' in space:
            space['max_depth'] = int(space['max_depth'])

        self.model = xgb.train(
            params=space,
            dtrain=self.dtrain,
            num_boost_round=n,
            evals=[(self.dtest, 'validation'), (self.dtrain, 'train')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )

        print('RSME: {rsme}'.format(rsme=self.rmse()))

    def cv(self, space, folds=5, early_stopping_rounds=None):
        n = int(space['num_boost_round'])

        if 'max_depth' in space:
            space['max_depth'] = int(space['max_depth'])

        results = xgb.cv(
            params=space,
            dtrain=self.dtrain,
            num_boost_round=n,
            seed=self.SEED,
            nfold=folds,
            early_stopping_rounds=early_stopping_rounds
        )
        rmse = results['test-rmse-mean'].min()
        print('RSME: {rsme}'.format(rsme=rmse))
        return rmse

    def get_model(self):
        if self.model is None:
            raise Exception('Model not trained. Run train() first.')
        return self.model

    def predict(self):
        self.predictions = self.model.predict(self.dtest)
        return self.predictions

    def mse(self):
        self.predict()
        return mean_squared_error(self.y_test, self.predictions)

    def rmse(self):
        self.predict()
        return np.sqrt(mean_squared_error(self.y_test, self.predictions))

    def shap_values(self):
        if self.model is None:
            raise Exception('Model not trained. Run train() first.')
        explainer = shap.Explainer(self.model)
        return explainer(self.X_test)
