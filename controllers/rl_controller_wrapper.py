import numpy as np
from controllers.base_controller import BaseController


class RLController(BaseController):
    """
    Placeholder RL controller wrapper.

    RL modeli geldiğinde burada:
    - model yükleme
    - predict / act
    mantığı eklenecek.
    """

    def __init__(self, model_path=None, model=None):
        self.model_path = model_path
        self.model = model

    def act(self, state):
        if self.model is None:
            raise NotImplementedError(
                "RLController henüz gerçek model ile bağlanmadı."
            )

        action = self.model.predict(state)
        return np.asarray(action, dtype=np.float32)