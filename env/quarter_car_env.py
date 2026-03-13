import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.road_profiles import (
    random_road,
    sinusoidal_road,
    bump_road,
    pothole_road,
)


class QuarterCarEnv(gym.Env):
    """
    Quarter-car active suspension environment.

    State:
        [z_s, z_u, z_s_dot, z_u_dot, z_r]

    Action:
        Continuous actuator force u  (N)

    Dynamics:
        m_s * z_s_ddot = -k_s(z_s-z_u) - c_s(z_s_dot-z_u_dot) + u
        m_u * z_u_ddot =  k_s(z_s-z_u) + c_s(z_s_dot-z_u_dot) - k_t(z_u-z_r) - u
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ms: float = 290.0,
        mu: float = 59.0,
        ks: float = 16812.0,
        cs: float = 1000.0,
        kt: float = 190000.0,
        dt: float = 0.01,
        episode_length: int = 2000,
        max_force: float = 3000.0,
        road_type: str = "random",
        road_amplitude: float = 0.01,
        road_frequency: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        # Physical parameters
        self.ms = ms
        self.mu = mu
        self.ks = ks
        self.cs = cs
        self.kt = kt

        # Simulation parameters
        self.dt = dt
        self.episode_length = episode_length
        self.max_force = max_force

        # Road parameters
        self.road_type = road_type
        self.road_amplitude = road_amplitude
        self.road_frequency = road_frequency

        # Internal state
        self.t = 0.0
        self.step_count = 0
        self.state = None
        self.prev_zs_ddot = 0.0

        # RNG
        self.rng = np.random.default_rng(seed)

        # State: [z_s, z_u, z_s_dot, z_u_dot, z_r]
        high = np.array([1.0, 1.0, 50.0, 50.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action: actuator force
        self.action_space = spaces.Box(
            low=np.array([-self.max_force], dtype=np.float32),
            high=np.array([self.max_force], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0.0
        self.step_count = 0
        self.prev_zs_ddot = 0.0

        # Initial state close to equilibrium
        z_s = 0.0
        z_u = 0.0
        z_s_dot = 0.0
        z_u_dot = 0.0
        z_r = self._road_profile(self.t)

        self.state = np.array([z_s, z_u, z_s_dot, z_u_dot, z_r], dtype=np.float32)

        return self.state, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        u = float(np.clip(action[0], -self.max_force, self.max_force))

        z_s, z_u, z_s_dot, z_u_dot, z_r = self.state

        # Quarter-car dynamics
        z_s_ddot = (
            -self.ks * (z_s - z_u)
            - self.cs * (z_s_dot - z_u_dot)
            + u
        ) / self.ms

        z_u_ddot = (
            self.ks * (z_s - z_u)
            + self.cs * (z_s_dot - z_u_dot)
            - self.kt * (z_u - z_r)
            - u
        ) / self.mu

        # Euler integration
        z_s_dot_new = z_s_dot + z_s_ddot * self.dt
        z_u_dot_new = z_u_dot + z_u_ddot * self.dt

        z_s_new = z_s + z_s_dot_new * self.dt
        z_u_new = z_u + z_u_dot_new * self.dt

        self.t += self.dt
        self.step_count += 1

        z_r_new = self._road_profile(self.t)

        self.state = np.array(
            [z_s_new, z_u_new, z_s_dot_new, z_u_dot_new, z_r_new],
            dtype=np.float32,
        )

        reward = self._compute_reward(
            z_s_ddot=z_s_ddot,
            z_s=z_s_new,
            z_u=z_u_new,
            u=u,
        )

        terminated = False
        truncated = self.step_count >= self.episode_length

        info = {
            "time": self.t,
            "body_acc": z_s_ddot,
            "wheel_acc": z_u_ddot,
            "control_force": u,
            "road_input": z_r_new,
            "suspension_deflection": z_s_new - z_u_new,
            "tire_deflection": z_u_new - z_r_new,
        }

        self.prev_zs_ddot = z_s_ddot

        return self.state, reward, terminated, truncated, info

    def _compute_reward(self, z_s_ddot: float, z_s: float, z_u: float, u: float) -> float:
        """
        Basic reward:
        - penalize body acceleration   -> comfort
        - penalize suspension travel   -> mechanical constraint
        - penalize control effort      -> actuator realism
        """
        suspension_deflection = z_s - z_u

        reward = (
            -1.0 * (z_s_ddot ** 2)
            -10.0 * (suspension_deflection ** 2)
            -1e-6 * (u ** 2)
        )

        return float(reward)

    def _road_profile(self, t: float) -> float:
        if self.road_type == "random":
            return random_road(self.rng, amplitude=self.road_amplitude)

        elif self.road_type == "sinusoidal":
            return sinusoidal_road(
                t,
                amplitude=self.road_amplitude,
                frequency=self.road_frequency,
                )

        elif self.road_type == "bump":
            return bump_road(t, amplitude=self.road_amplitude)

        elif self.road_type == "pothole":
            return pothole_road(t, amplitude=self.road_amplitude)

        return 0.0

    def render(self, mode="human"):
        # Placeholder
        pass

    def close(self):
        pass