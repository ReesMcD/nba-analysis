from sklearn.model_selection import TimeSeriesSplit
from tools.modeling.modeler import Modeler
import numpy as np


class TimeSeriesModeler(Modeler):
    def __init__(self, model, X_train=None, X_test=None, y_train=None, y_test=None, time_series_split=TimeSeriesSplit(),
                 SEED=1):
        super().__init__(model, X_train, X_test, y_train, y_test, SEED)
        self.time_series_split = time_series_split
        self.predictions = None
        self.ts_rmse_arr = []

    def ts_train(self, params, df, target_col):
        for train_index, val_index in self.time_series_split.split(df):
            train = df.iloc[train_index]
            test = df.iloc[val_index]

            self.X_train, self.y_train = train.drop(target_col, axis=1), train[target_col]
            self.X_test, self.y_test = test.drop(target_col, axis=1), test[target_col]

            self.train(params)
            self.ts_rmse_arr.append(self.rmse())

        print('RSME: {rsme}'.format(rsme=self.ts_rmse()))

    def ts_rmse(self):
        return np.average(self.ts_rmse_arr)