from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from shaphypetune import BoostRFA
from xgboost import XGBRegressor
from hyperopt import Trials

import numpy as np

RMSE = 'rmse'
SHAP_IMPORTANCES = 'shap_importances'


def _rmse(y_test, predictions):
    return np.sqrt(mean_squared_error(y_test, predictions))


# TODO: Allow Modeler to accept feature mask to allow for a re train after feature selection
class TimeSeriesModeler:
    def __init__(self, model: XGBRegressor, df, target, cv=TimeSeriesSplit(), score_eval=RMSE, SEED=1):
        self._model = model
        self._df = df
        self._target = target
        self._cv = cv
        self._score_eval = score_eval
        self._SEED = SEED

        self._split_scores = []
        self._tuned_params = []
        self._tuned_features = []
        self._score = None
        self._params = None
        self._space = None
        self._rsme = None
        self._feature_mask = None

    def train(self, params):
        self._params = params
        self._time_series_loop(self._train)
        self._calculate_score()

    def hypertune_and_feature_selection(self, space):
        temp_df = self._df

        self._space = space
        self._time_series_loop(self._hypertune_and_feature_selection)

        columns = self._df.drop(self._target, axis=1).columns
        potential_features = list(map(lambda f: columns[f], self._tuned_features))
        potential_params = self._tuned_params

        # TODO: Clean this up
        best_scores = []
        best_features = []
        best_params = []

        for features in potential_features:
            for param in potential_params:
                self._df = temp_df

                mask = [self._target] + features.tolist()
                self._df = self._df[mask]

                self.train(param)
                best_scores.append(self._score)
                best_features.append(features)
                best_params.append(param)

        self._df = temp_df
        self._score = min(best_scores)

        i = best_scores.index(self._score)
        return best_features[i], best_params[i]

    def _time_series_loop(self, func):
        for i, (train_index, val_index) in enumerate(self._cv.split(self._df)):
            train = self._df.iloc[train_index]
            test = self._df.iloc[val_index]

            X_train, y_train = train.drop(self._target, axis=1), train[self._target]
            X_test, y_test = test.drop(self._target, axis=1), test[self._target]
            func(X_train, y_train, X_test, y_test)
            print('Cross Validation Iteration: {i}'.format(i=i + 1))

    def _train(self, X_train, y_train, X_test, y_test):
        self._model.set_params(**self._params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100,
        )

        self._update_score(X_test, y_test)

    def _hypertune_and_feature_selection(self, X_train, y_train, X_test, y_test):
        # TODO: Make paramaters configurable
        model = BoostRFA(
            self._model,
            param_grid=self._space,
            min_features_to_select=1,
            step=1,
            n_iter=250,
            sampling_seed=self._SEED,
            importance_type=SHAP_IMPORTANCES,
            train_importance=False,
        )

        model.fit(
            X_train,
            y_train,
            trials=Trials(),
            eval_set=[(X_test, y_test)],
            verbose=0,
        )

        self._tuned_params.append(model.best_params_)
        self._tuned_features.append(model.support_)

    def _calculate_score(self):
        self._score = np.average(self._split_scores)

    def _update_score(self, X_test, y_test):
        predictions = self._predict(X_test)

        if self._score_eval == RMSE:
            self._split_scores.append(_rmse(y_test, predictions))

    def _predict(self, X_test):
        return self._model.predict(X_test)

    @property
    def score(self):
        return self._score
