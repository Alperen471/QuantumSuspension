from experiments.run_experiment import summarize_scenario
from analysis.result_annotations import add_metric_labels
from analysis.results_io import save_summary_json, append_summary_csv


def evaluate_controller_on_road(
    controller_type: str,
    controller_params: dict,
    road_config: dict,
    dt: float,
    episode_length: int,
    seed: int,
    save_results: bool = True,
    verbose: bool = True,
):
    summary = summarize_scenario(
        controller_type=controller_type,
        controller_params=controller_params,
        road_type=road_config["road_type"],
        road_amplitude=road_config["road_amplitude"],
        road_frequency=road_config["road_frequency"],
        episode_length=episode_length,
        dt=dt,
        seed=seed,
    )

    summary = add_metric_labels(summary)

    if save_results:
        json_path = save_summary_json(summary)
        csv_path = append_summary_csv(summary)

        if verbose:
            print(f"JSON saved to: {json_path}")
            print(f"CSV updated : {csv_path}")

    return summary


def evaluate_controller(
    controller_type: str,
    controller_params: dict,
    roads: list[dict],
    seeds: list[int],
    dt: float,
    episode_length: int,
    save_results: bool = True,
    verbose: bool = True,
):
    all_summaries = []

    for seed in seeds:
        for road_config in roads:
            summary = evaluate_controller_on_road(
                controller_type=controller_type,
                controller_params=controller_params,
                road_config=road_config,
                dt=dt,
                episode_length=episode_length,
                seed=seed,
                save_results=save_results,
                verbose=verbose,
            )
            all_summaries.append(summary)

    return all_summaries


def evaluate_multiple_controllers(
    controller_configs: list[dict],
    roads: list[dict],
    seeds: list[int],
    dt: float,
    episode_length: int,
    save_results: bool = True,
    verbose: bool = True,
):
    all_summaries = []

    for ctrl_cfg in controller_configs:
        controller_type = ctrl_cfg["controller_type"]
        controller_params = ctrl_cfg.get("controller_params", {})

        if verbose:
            print("\n" + "=" * 100)
            print(f"Evaluating controller: {controller_type}")
            print(f"Controller params    : {controller_params}")
            print("=" * 100)

        summaries = evaluate_controller(
            controller_type=controller_type,
            controller_params=controller_params,
            roads=roads,
            seeds=seeds,
            dt=dt,
            episode_length=episode_length,
            save_results=save_results,
            verbose=verbose,
        )

        all_summaries.extend(summaries)

    return all_summaries