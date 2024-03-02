from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import shap


class Modeler:
    def __init__(self, model: XGBRegressor, X_train=None, X_test=None, y_train=None, y_test=None, SEED=1):
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.SEED = SEED
        self.predictions = None

    def train(self, space):

        if 'n_estimators' in space:
            space['n_estimators'] = int(space['n_estimators'])

        if 'max_depth' in space:
            space['max_depth'] = int(space['max_depth'])

        self.model.set_params(**space)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=100,
        )

        self.predictions = self.model.predict(self.X_test)
        print('RSME: {rsme}'.format(rsme=self.rmse()))

    def get_model(self):
        return self.model

    def predict(self):
        self.predictions = self.model.predict(self.X_test)
        return self.predictions

    def mse(self):
        self.predict()
        return mean_squared_error(self.y_test, self.predictions)

    def rmse(self):
        self.predict()
        return np.sqrt(mean_squared_error(self.y_test, self.predictions))

    def shap_values(self):
        explainer = shap.Explainer(self.model)
        return explainer(self.X_test)

    def set_X_test(self, X_test):
        self.X_test = X_test
        return self

    def set_y_test(self, y_test):
        self.y_test = y_test
        return self

    def set_X_train(self, X_train):
        self.X_train = X_train
        return self

    def set_y_train(self, y_train):
        self.y_train = y_train
        return self

