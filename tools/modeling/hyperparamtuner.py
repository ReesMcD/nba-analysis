from tools.modeling.modeler import Modeler
from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
import numpy as np


class HyperParamTuner:
    def __init__(self, modeler: Modeler, space, SEED=1):
        self.modeler = modeler
        self.space = space
        self.SEED = SEED

    def objective(self, params):
        self.modeler.train(params)

        score = self.modeler.rmse()
        return {'loss': score, 'status': STATUS_OK, 'model': self.modeler.get_model()}

    def optimize(self, max_evals):
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    rstate=np.random.default_rng(self.SEED)
                    )
        return space_eval(self.space, best)
