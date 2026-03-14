import numpy as np

from env.quarter_car_env import QuarterCarEnv
from controllers.passive_controller import PassiveController
from controllers.skyhook_controller import SkyhookController
from controllers.rl_controller_wrapper import RLController


from analysis.time_domain_metrics import compute_time_domain_metrics
from analysis.frequency_domain_metrics import compute_frequency_domain_metrics
from analysis.result_annotations import add_metric_labels
from analysis.results_io import save_summary_json, append_summary_csv


def build_controller(controller_type: str, controller_params: dict | None = None):
    controller_params = controller_params or {}

    if controller_type == "passive":
        return PassiveController()

    elif controller_type == "skyhook":
        c_sky = controller_params.get("c_sky", 4000.0)
        max_force = controller_params.get("max_force", 3000.0)
        return SkyhookController(c_sky=c_sky, max_force=max_force)
    
    elif controller_type == "rl":
        model_path = controller_params.get("model_path",None)
        model = controller_params.get("model",None)
        return RLController(model_path=model_path,model=model)

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def run_single_experiment(
    controller_type="passive",
    controller_params=None,
    road_type="random",
    road_amplitude=0.01,
    road_frequency=1.0,
    episode_length=2000,
    dt=0.01,
    seed=None,
):
    env = QuarterCarEnv(
        road_type=road_type,
        road_amplitude=road_amplitude,
        road_frequency=road_frequency,
        episode_length=episode_length,
        dt=dt,
        seed=seed,
    )

    controller = build_controller(controller_type, controller_params)

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
    controller_type,
    controller_params,
    road_type,
    road_amplitude,
    road_frequency,
    episode_length,
    dt,
    seed=None,
):
    controller_params = controller_params or {}
    
    results = run_single_experiment(
        controller_type=controller_type,
        controller_params=controller_params,
        road_type=road_type,
        road_amplitude=road_amplitude,
        road_frequency=road_frequency,
        episode_length=episode_length,
        dt=dt,
        seed=seed,
    )

    time_metrics = compute_time_domain_metrics(results)
    freq_metrics = compute_frequency_domain_metrics(results, dt=dt)

    summary = {
        "controller_type": controller_type,
        "controller_params": controller_params or {},
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
        "iso_weighted_rms_body_acc": freq_metrics["iso_weighted_rms_body_acc"],
        "seed":seed,
    }

    return summary


def pretty_print_summary(summary: dict):
    labels = summary["labels"]

    print("=" * 90)
    print(f"Controller type      : {summary['controller_type']}")
    print(f"Controller params    : {summary.get('controller_params', {})}"),
    print(f"Road type            : {summary['road_type']}")
    print(f"Road amplitude       : {summary['road_amplitude']:.6f}")
    print(f"Road frequency       : {summary['road_frequency']:.6f}")
    print(f"Number of steps      : {summary['num_steps']}")
    print(f"Mean reward          : {summary['mean_reward']:.6f}")

    print("\nTime-domain metrics")
    print(f"  rms_body_acc               : {summary['rms_body_acc']:.6f} ({labels['rms_body_acc']})")
    print(f"  max_abs_body_acc           : {summary['max_abs_body_acc']:.6f} ({labels['max_abs_body_acc']})")
    print(f"  rms_wheel_acc              : {summary['rms_wheel_acc']:.6f}")
    print(f"  max_abs_wheel_acc          : {summary['max_abs_wheel_acc']:.6f}")
    print(f"  rms_control_force          : {summary['rms_control_force']:.6f} ({labels['rms_control_force']})")
    print(f"  max_abs_control_force      : {summary['max_abs_control_force']:.6f}")
    print(
        f"  rms_suspension_deflection  : "
        f"{summary['rms_suspension_deflection']:.6f} ({labels['rms_suspension_deflection']})"
    )
    print(f"  max_abs_suspension_deflection: {summary['max_abs_suspension_deflection']:.6f}")
    print(
        f"  rms_tire_deflection        : "
        f"{summary['rms_tire_deflection']:.6f} ({labels['rms_tire_deflection']})"
    )
    print(f"  max_abs_tire_deflection    : {summary['max_abs_tire_deflection']:.6f}")

    print("\nFrequency-domain metrics")
    print(
        f"  total_psd_energy           : "
        f"{summary['total_psd_energy']:.6f} ({labels['total_psd_energy']})"
    )
    print(
        f"  band_1_3Hz_energy          : "
        f"{summary['band_1_3Hz_energy']:.6f} ({labels['band_1_3Hz_energy']})"
    )
    print(f"  band_0_5_5Hz_energy        : {summary['band_0_5_5Hz_energy']:.6f}")
    print(f"  dominant_frequency         : {summary['dominant_frequency']:.6f}")
    print("=" * 90)
    print(
    f"  iso_weighted_rms_body_acc  : "
    f"{summary['iso_weighted_rms_body_acc']:.6f} "
    f"({labels['iso_weighted_rms_body_acc']})"
    )   


def print_compact_table(all_summaries: list[dict]):
    print("\n" + "=" * 150)
    print("CONTROLLER × ROAD BENCHMARK TABLE")
    print("=" * 150)

    header = (
        f"{'controller':<12}"
        f"{'road':<12}"
        f"{'rms_body':>12}"
        f"{'body_lbl':>12}"
        f"{'rms_force':>12}"
        f"{'force_lbl':>12}"
        f"{'1-3Hz':>12}"
        f"{'band_lbl':>12}"
        f"{'total_psd':>14}"
        f"{'psd_lbl':>12}"
    )
    print(header)
    print("-" * 150)

    for s in all_summaries:
        row = (
            f"{s['controller_type']:<12}"
            f"{s['road_type']:<12}"
            f"{s['rms_body_acc']:>12.6f}"
            f"{s['labels']['rms_body_acc']:>12}"
            f"{s['rms_control_force']:>12.6f}"
            f"{s['labels']['rms_control_force']:>12}"
            f"{s['band_1_3Hz_energy']:>12.6f}"
            f"{s['labels']['band_1_3Hz_energy']:>12}"
            f"{s['total_psd_energy']:>14.6f}"
            f"{s['labels']['total_psd_energy']:>12}"
        )
        print(row)

    print("=" * 150)


def run_controller_road_benchmark():
    dt = 0.01
    episode_length = 2000

    controller_configs = [
        {"controller_type": "passive","controller_params":{}},
        {"controller_type": "skyhook",
         "controller_params":{
             "c_sky": 4000.0,
             "max_force":3000.0,
         }},
    ]

    road_configs = [
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

    for controller_cfg in controller_configs:
        for road_cfg in road_configs:
            summary = summarize_scenario(
                controller_type=controller_cfg["controller_type"],
                controller_params=controller_cfg.get("controller_params",{}),
                road_type=road_cfg["road_type"],
                road_amplitude=road_cfg["road_amplitude"],
                road_frequency=road_cfg["road_frequency"],
                episode_length=episode_length,
                dt=dt,
                seed=None,
            )

            summary = add_metric_labels(summary)

            pretty_print_summary(summary)

            json_path = save_summary_json(summary)
            csv_path = append_summary_csv(summary)

            print(f"JSON saved to: {json_path}")
            print(f"CSV updated : {csv_path}")

            all_summaries.append(summary)

    print_compact_table(all_summaries)

    return all_summaries

def print_skyhook_tuning_table(all_summaries: list[dict]):
    print("\n" + "=" * 170)
    print("SKYHOOK GAIN TUNING TABLE")
    print("=" * 170)

    header = (
        f"{'road':<12}"
        f"{'c_sky':>10}"
        f"{'rms_body':>12}"
        f"{'body_lbl':>12}"
        f"{'rms_force':>12}"
        f"{'force_lbl':>12}"
        f"{'1-3Hz':>12}"
        f"{'band_lbl':>12}"
        f"{'total_psd':>14}"
        f"{'psd_lbl':>12}"
        f"{'max_body':>12}"
    )
    print(header)
    print("-" * 170)

    for s in all_summaries:
        c_sky = s.get("controller_params", {}).get("c_sky", "")
        row = (
            f"{s['road_type']:<12}"
            f"{c_sky:>10}"
            f"{s['rms_body_acc']:>12.6f}"
            f"{s['labels']['rms_body_acc']:>12}"
            f"{s['rms_control_force']:>12.6f}"
            f"{s['labels']['rms_control_force']:>12}"
            f"{s['band_1_3Hz_energy']:>12.6f}"
            f"{s['labels']['band_1_3Hz_energy']:>12}"
            f"{s['total_psd_energy']:>14.6f}"
            f"{s['labels']['total_psd_energy']:>12}"
            f"{s['max_abs_body_acc']:>12.6f}"
        )
        print(row)

    print("=" * 170)


def run_skyhook_gain_tuning():
    dt = 0.01
    episode_length = 2000

    gain_values = [1000.0, 2000.0, 2500.0, 4000.0, 6000.0]

    road_configs = [
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
    ]

    all_summaries = []

    for gain in gain_values:
        for road_cfg in road_configs:
            summary = summarize_scenario(
                controller_type="skyhook",
                controller_params={
                    "c_sky": gain,
                    "max_force": 3000.0,
                },
                road_type=road_cfg["road_type"],
                road_amplitude=road_cfg["road_amplitude"],
                road_frequency=road_cfg["road_frequency"],
                episode_length=episode_length,
                dt=dt,
            )

            summary = add_metric_labels(summary)

            pretty_print_summary(summary)

            json_path = save_summary_json(summary)
            csv_path = append_summary_csv(summary)

            print(f"JSON saved to: {json_path}")
            print(f"CSV updated : {csv_path}")

            all_summaries.append(summary)

    print_skyhook_tuning_table(all_summaries)

    return all_summaries

if __name__ == "__main__":
    run_skyhook_gain_tuning()