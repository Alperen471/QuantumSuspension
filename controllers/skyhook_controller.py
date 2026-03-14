import numpy as np
from controllers.base_controller import BaseController


class SkyhookController(BaseController):
    """
    Simple skyhook-like active damping controller.
    u = -c_sky * z_s_dot
    """

    def __init__(self, c_sky: float = 4000.0, max_force: float = 3000.0):
        self.c_sky = c_sky
        self.max_force = max_force

    def act(self, state):
        z_s, z_u, z_s_dot, z_u_dot, z_r = state

        # Skyhook control law
        u = -self.c_sky * z_s_dot

        # actuator saturation
        u = np.clip(u, -self.max_force, self.max_force)

        return np.array([u], dtype=np.float32)