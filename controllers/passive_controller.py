import numpy as np
from controllers.base_controller import BaseController


class PassiveController(BaseController):
    def act(self, state):
        return np.array([0.0], dtype=np.float32)