import numpy as np


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def max_abs(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def extract_time_series(results: dict):
    """
    results:
        {
            "states": np.ndarray,
            "rewards": np.ndarray,
            "infos": list[dict]
        }
    """
    infos = results.get("infos", [])
    rewards = np.asarray(results.get("rewards", []), dtype=np.float64)

    body_acc = np.array([info["body_acc"] for info in infos], dtype=np.float64)
    wheel_acc = np.array([info["wheel_acc"] for info in infos], dtype=np.float64)
    control_force = np.array([info["control_force"] for info in infos], dtype=np.float64)
    road_input = np.array([info["road_input"] for info in infos], dtype=np.float64)
    suspension_deflection = np.array(
        [info["suspension_deflection"] for info in infos], dtype=np.float64
    )
    tire_deflection = np.array(
        [info["tire_deflection"] for info in infos], dtype=np.float64
    )
    time = np.array([info["time"] for info in infos], dtype=np.float64)

    return {
        "time": time,
        "body_acc": body_acc,
        "wheel_acc": wheel_acc,
        "control_force": control_force,
        "road_input": road_input,
        "suspension_deflection": suspension_deflection,
        "tire_deflection": tire_deflection,
        "rewards": rewards,
    }


def compute_time_domain_metrics(results: dict) -> dict:
    series = extract_time_series(results)

    metrics = {
        "rms_body_acc": rms(series["body_acc"]),
        "max_abs_body_acc": max_abs(series["body_acc"]),
        "rms_wheel_acc": rms(series["wheel_acc"]),
        "max_abs_wheel_acc": max_abs(series["wheel_acc"]),
        "rms_control_force": rms(series["control_force"]),
        "max_abs_control_force": max_abs(series["control_force"]),
        "rms_suspension_deflection": rms(series["suspension_deflection"]),
        "max_abs_suspension_deflection": max_abs(series["suspension_deflection"]),
        "rms_tire_deflection": rms(series["tire_deflection"]),
        "max_abs_tire_deflection": max_abs(series["tire_deflection"]),
        "mean_reward": float(np.mean(series["rewards"])) if series["rewards"].size > 0 else 0.0,
    }

    return metrics