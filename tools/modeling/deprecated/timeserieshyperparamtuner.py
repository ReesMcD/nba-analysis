from tools.modeling.deprecated.hyperparamtuner import HyperParamTuner
from tools.modeling.deprecated.timeseriesmodeler import TimeSeriesModeler
from hyperopt import STATUS_OK


class TimeSeriesHyperParamTuner(HyperParamTuner):
    def __init__(self, modeler: TimeSeriesModeler, space, SEED=1):
        super().__init__(modeler, space, SEED)
        self.df = None
        self.target_col = None

    def ts_optimize(self, max_evals, df, taget_col):
        self.df = df
        self.target_col = taget_col
        return self.optimize(max_evals)

    def objective(self, params):
        self.modeler.ts_train(params, self.df, self.target_col)
        return {'loss': self.modeler.ts_rmse(), 'status': STATUS_OK, 'model': self.modeler.get_model()}
