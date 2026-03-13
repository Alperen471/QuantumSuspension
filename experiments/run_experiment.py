import numpy as np

from env.quarter_car_env import QuarterCarEnv
from controllers.passive_controller import PassiveController
from analysis.time_domain_metrics import compute_time_domain_metrics
from analysis.frequency_domain_metrics import compute_frequency_domain_metrics


def run_single_experiment(
    road_type: str = "random",
    road_amplitude: float = 0.01,
    road_frequency: float = 1.0,
    episode_length: int = 2000,
    dt: float = 0.01,
):
    env = QuarterCarEnv(
        road_type=road_type,
        road_amplitude=road_amplitude,
        road_frequency=road_frequency,
        episode_length=episode_length,
        dt=dt,
    )

    controller = PassiveController()

    state, _ = env.reset()

    state_log = []
    reward_log = []
    info_log = []

    done = False
    truncated = False

    while not (done or truncated):
        action = controller.act(state)
        next_state, reward, done, truncated, info = env.step(action)

        state_log.append(next_state.copy())
        reward_log.append(reward)
        info_log.append(info)

        state = next_state

    env.close()

    return {
        "states": np.array(state_log),
        "rewards": np.array(reward_log),
        "infos": info_log,
    }


def summarize_scenario(
    road_type: str,
    road_amplitude: float,
    road_frequency: float,
    episode_length: int,
    dt: float,
):
    results = run_single_experiment(
        road_type=road_type,
        road_amplitude=road_amplitude,
        road_frequency=road_frequency,
        episode_length=episode_length,
        dt=dt,
    )

    time_metrics = compute_time_domain_metrics(results)
    freq_metrics = compute_frequency_domain_metrics(results, dt=dt)

    summary = {
        "road_type": road_type,
        "road_amplitude": road_amplitude,
        "road_frequency": road_frequency,
        "num_steps": len(results["states"]),
        "mean_reward": float(np.mean(results["rewards"])) if len(results["rewards"]) > 0 else 0.0,
        "rms_body_acc": time_metrics["rms_body_acc"],
        "max_abs_body_acc": time_metrics["max_abs_body_acc"],
        "rms_wheel_acc": time_metrics["rms_wheel_acc"],
        "max_abs_wheel_acc": time_metrics["max_abs_wheel_acc"],
        "rms_control_force": time_metrics["rms_control_force"],
        "max_abs_control_force": time_metrics["max_abs_control_force"],
        "rms_suspension_deflection": time_metrics["rms_suspension_deflection"],
        "max_abs_suspension_deflection": time_metrics["max_abs_suspension_deflection"],
        "rms_tire_deflection": time_metrics["rms_tire_deflection"],
        "max_abs_tire_deflection": time_metrics["max_abs_tire_deflection"],
        "total_psd_energy": freq_metrics["total_psd_energy"],
        "band_1_3Hz_energy": freq_metrics["band_1_3Hz_energy"],
        "band_0_5_5Hz_energy": freq_metrics["band_0_5_5Hz_energy"],
        "dominant_frequency": freq_metrics["dominant_frequency"],
    }

    return summary


def print_scenario_summary(summary: dict):
    print("=" * 72)
    print(f"Road type            : {summary['road_type']}")
    print(f"Road amplitude       : {summary['road_amplitude']:.6f}")
    print(f"Road frequency       : {summary['road_frequency']:.6f}")
    print(f"Number of steps      : {summary['num_steps']}")
    print(f"Mean reward          : {summary['mean_reward']:.6f}")

    print("\nTime-domain metrics")
    print(f"  rms_body_acc               : {summary['rms_body_acc']:.6f}")
    print(f"  max_abs_body_acc           : {summary['max_abs_body_acc']:.6f}")
    print(f"  rms_wheel_acc              : {summary['rms_wheel_acc']:.6f}")
    print(f"  max_abs_wheel_acc          : {summary['max_abs_wheel_acc']:.6f}")
    print(f"  rms_control_force          : {summary['rms_control_force']:.6f}")
    print(f"  max_abs_control_force      : {summary['max_abs_control_force']:.6f}")
    print(f"  rms_suspension_deflection  : {summary['rms_suspension_deflection']:.6f}")
    print(f"  max_abs_suspension_deflection: {summary['max_abs_suspension_deflection']:.6f}")
    print(f"  rms_tire_deflection        : {summary['rms_tire_deflection']:.6f}")
    print(f"  max_abs_tire_deflection    : {summary['max_abs_tire_deflection']:.6f}")

    print("\nFrequency-domain metrics")
    print(f"  total_psd_energy           : {summary['total_psd_energy']:.6f}")
    print(f"  band_1_3Hz_energy          : {summary['band_1_3Hz_energy']:.6f}")
    print(f"  band_0_5_5Hz_energy        : {summary['band_0_5_5Hz_energy']:.6f}")
    print(f"  dominant_frequency         : {summary['dominant_frequency']:.6f}")
    print("=" * 72)


def print_compact_table(all_summaries: list[dict]):
    print("\n" + "=" * 120)
    print("COMPACT BENCHMARK TABLE")
    print("=" * 120)
    header = (
        f"{'road':<12}"
        f"{'rms_body':>12}"
        f"{'max_body':>12}"
        f"{'rms_susp':>12}"
        f"{'rms_tire':>12}"
        f"{'1-3Hz':>12}"
        f"{'total_psd':>14}"
        f"{'dom_freq':>12}"
    )
    print(header)
    print("-" * 120)

    for s in all_summaries:
        row = (
            f"{s['road_type']:<12}"
            f"{s['rms_body_acc']:>12.6f}"
            f"{s['max_abs_body_acc']:>12.6f}"
            f"{s['rms_suspension_deflection']:>12.6f}"
            f"{s['rms_tire_deflection']:>12.6f}"
            f"{s['band_1_3Hz_energy']:>12.6f}"
            f"{s['total_psd_energy']:>14.6f}"
            f"{s['dominant_frequency']:>12.6f}"
        )
        print(row)

    print("=" * 120)


def run_scenario_suite():
    dt = 0.01
    episode_length = 2000

    scenario_configs = [
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

    all_summaries = []

    for cfg in scenario_configs:
        summary = summarize_scenario(
            road_type=cfg["road_type"],
            road_amplitude=cfg["road_amplitude"],
            road_frequency=cfg["road_frequency"],
            episode_length=episode_length,
            dt=dt,
        )
        all_summaries.append(summary)
        print_scenario_summary(summary)

    print_compact_table(all_summaries)

    return all_summaries


if __name__ == "__main__":
    run_scenario_suite()