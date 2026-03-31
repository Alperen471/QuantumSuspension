EVALUATION_CONFIG = {
    "dt": 0.01,
    "episode_length": 2000,
    "seeds": [1, 2, 3, 4, 5],
}

TEST_ROADS = [
    {
        "road_type": "random",
        "road_amplitude": 0.01,
        "road_frequency": 1.0,
    },
    {
        "road_type": "sinusoidal",
        "road_amplitude": 0.01,
        "road_frequency": 1.0,
    },
    {
        "road_type": "bump",
        "road_amplitude": 0.02,
        "road_frequency": 1.0,
    },
    {
        "road_type": "pothole",
        "road_amplitude": 0.02,
        "road_frequency": 1.0,
    },
]

CONTROLLER_CONFIGS = [
    {
        "controller_type": "passive",
        "controller_params": {},
    },
    {
        "controller_type": "skyhook",
        "controller_params": {
            "c_sky": 4000.0,
            "max_force": 3000.0,
        },
    },
    # RL geldiğinde buraya eklenecek
    # {
    #     "controller_type": "rl",
    #     "controller_params": {
    #         "model_path": "models/td3_best.pt"
    #     },
    # },
]