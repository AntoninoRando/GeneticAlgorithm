import argparse
import random
from datetime import datetime
from typing import TypedDict

from scheduler import SchedulerGP, FitnessEvaluator, SimulationSnapshot, buildTerminalRegistry
from data_processing.data_factory import create_jobs_from_csv, create_satellites_from_csv
from data_processing.csv_preprocessing import load_world_paths
from metrics_tracker import GenerationMetrics, MetricsCollector
from snapshot_inspector import print_simulation_snapshot_summary, save_simulation_snapshot


class GAConfig(TypedDict):
    # GA parameters
    generations: int
    population_size: int
    max_depth: int
    hoist_rate: float
    convergence_threshold: int
    csv_row_limit: int
    # Metrics
    metrics_dir: str
    metrics_prefix: str
    metrics_flush_every: int
    collect_metrics: bool
    # Snapshots
    inspect_snapshots: bool
    save_snapshots: bool
    snapshots_dir: str
    snapshots_prefix: str
    # Miscellaneous
    print_every: int


def default_config() -> GAConfig:
    return GAConfig(
        generations=200,
        population_size=100,
        max_depth=7,
        hoist_rate=0.05,
        convergence_threshold=10,
        csv_row_limit=100,
        metrics_dir="metrics",
        metrics_prefix="",
        metrics_flush_every=1,
        collect_metrics=True,
        inspect_snapshots=False,
        save_snapshots=False,
        snapshots_dir="snapshots",
        snapshots_prefix="",
        print_every=50,
    )



#region ARGS
parser = argparse.ArgumentParser(description="Run GP Scheduler")


#region GA parameters
parser.add_argument("--population", type=int, default=100, help="Population size")
parser.add_argument("--generations", type=int, default=200, help="Number of generations")
parser.add_argument("--max-depth", type=int, default=7,
                    help="Maximum expression tree depth. Shallower trees are faster and simpler "
                         "(e.g. --max-depth 4). Max nodes ≈ 2^(depth+1), so depth 7 → ≤256 nodes.")
parser.add_argument("--hoist-rate", type=float, default=0.05,
                    help="Probability [0..1] that each new offspring undergoes a hoist mutation, "
                         "which replaces the whole tree with one of its own subtrees to fight bloat "
                         "(e.g. --hoist-rate 0.20 for aggressive pruning).")
parser.add_argument("--convergence-threshold", type=int, default=10, 
                    help="Number of generations without fitness improvement before regenerating satellites and jobs (G).")
parser.add_argument("--csv-row-limit", type=int, default=100,
                    help="Maximum number of rows to load from the CSV file for satellites and jobs. "
                         "Set to 0 or a negative number to load all rows (e.g. --csv-row-limit 50).")
#endregion



#region Metrics
parser.add_argument("--metrics-dir", type=str, default="metrics",
                    help="Directory where metrics artifacts are written.")
parser.add_argument("--metrics-prefix", type=str, default="",
                    help="Output prefix for metrics files. Defaults to a timestamp.")
parser.add_argument("--metrics-flush-every", type=int, default=1,
                    help="How often (in generations) to refresh the NumPy metrics dump.")
parser.add_argument("--disable-metrics", action="store_true",
                    help="Disable metrics collection and graph generation.")
#endregion



#region Snapshots
parser.add_argument("--inspect-snapshots", action="store_true",
                    help="Print readable summaries for initial/reset/final simulation snapshots.")
parser.add_argument("--save-snapshots", action="store_true",
                    help="Persist inspected snapshots as JSON files.")
parser.add_argument("--snapshots-dir", type=str, default="snapshots",
                    help="Directory where snapshot JSON files are written.")
parser.add_argument("--snapshots-prefix", type=str, default="",
                    help="Output prefix for snapshot JSON files. Defaults to a timestamp.")
#endregion


#region Miscellaneous
parser.add_argument("--no-start-prompt", action="store_true",
                    help="Start immediately without waiting for Enter.")
parser.add_argument("--print-every", type=int, default=50, help="Print progress every N generations")
#endregion


#region Config file
parser.add_argument("-c", "--config", type=str, default=None,
                    help="Name of a run configuration (without extension) in the runConfigurations/ "
                         "directory, e.g. '--config default' loads runConfigurations/default.json. "
                         "Known keys override the built-in defaults; unrecognized keys are ignored. "
                         "Explicit CLI flags always take precedence over the JSON values.")
#endregion
#endregion



#region Main loop
def run_genetic_algorithm(config: GAConfig):
    generations        = config["generations"]
    population_size    = config["population_size"]
    max_depth          = config["max_depth"]
    hoist_rate         = config["hoist_rate"]
    convergence_threshold = config["convergence_threshold"]
    csv_row_limit      = config["csv_row_limit"]
    metrics_dir        = config["metrics_dir"]
    metrics_prefix     = config["metrics_prefix"]
    metrics_flush_every = config["metrics_flush_every"]
    collect_metrics    = config["collect_metrics"]
    inspect_snapshots  = config["inspect_snapshots"]
    save_snapshots     = config["save_snapshots"]
    snapshots_dir      = config["snapshots_dir"]
    snapshots_prefix   = config["snapshots_prefix"]
    print_every        = config["print_every"]

    if generations <= 0:
        print("No generations requested; nothing to optimize.")
        return



    #region > Initialization
    # Pre-processed worlds: one CSV per simulation time window, with arrival
    # times already normalized relatively (see data_processing.csv_preprocessing).
    # Each GA "world reset" advances to the next world so the population faces a
    # genuinely different scenario rather than a re-sample of a single CSV.
    world_paths = [str(path) for path in load_world_paths("data/worlds")]
    if world_paths:
        print(f"Loaded {len(world_paths)} pre-processed world(s) from data/worlds/.")
    else:
        world_paths = ["data/example.csv"]
        print("No pre-processed worlds in data/worlds/; falling back to "
              "data/example.csv. Run "
              "`python -m data_processing.csv_preprocessing.world_splitter` "
              "to generate worlds.")

    def select_world_path(world_index: int) -> str:
        return world_paths[world_index % len(world_paths)]

    initial_job_limit = csv_row_limit
    current_minute = 0.0
    world_rng = random.Random()
    snapshot_prefix_value = snapshots_prefix.strip() or datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")

    def inspect_snapshot(active_snapshot: SimulationSnapshot, generation: int, reset_count: int, label: str):
        if not (inspect_snapshots or save_snapshots):
            return

        print(f"\n[{label}]")
        if inspect_snapshots:
            print_simulation_snapshot_summary(active_snapshot)

        if save_snapshots:
            snapshot_path = save_simulation_snapshot(
                active_snapshot,
                output_dir=snapshots_dir,
                prefix=snapshot_prefix_value,
                generation=generation,
                reset_count=reset_count,
            )
            print(f"Snapshot JSON: {snapshot_path}")

    def build_world(world_path: str, job_limit: int):
        world_seed = world_rng.randrange(0, 2_147_483_647)
        # A non-positive limit means "load the whole world" (matches the
        # --csv-row-limit help text); pre-split worlds are small by design.
        effective_limit = job_limit if (job_limit and job_limit > 0) else None
        # Jobs first: arrivals are relative (earliest task at minute 0), so the
        # snapshot's "now" is derived from them rather than fixed in advance.
        jobs_local = create_jobs_from_csv(
            world_path,
            limit=effective_limit,
            random_sample=False,
            sample_seed=world_seed,
        )
        # Evaluate at the end of the arrival window so every task has already
        # arrived (the evaluator drops jobs whose arrivalTime exceeds the
        # snapshot minute).  Response time stays = completion - arrivalTime.
        minute = float(max((job.arrivalTime for job in jobs_local), default=0))
        satellites_local = create_satellites_from_csv(
            world_path,
            limit=effective_limit,
            seed=world_seed,
            random_sample=False,
            sample_seed=world_seed,
            snapshot_minute=minute,
        )
        return satellites_local, jobs_local, minute, world_seed

    satellites, jobs, current_minute, _ = build_world(select_world_path(0), initial_job_limit)

    genetic_algorithm = SchedulerGP(satellites, jobs)
    genetic_algorithm.initialize(populationSize=population_size, maxDepth=max_depth,
                                 hoistRate=hoist_rate)
    fitness_evaluator = FitnessEvaluator(buildTerminalRegistry())
    snapshot = SimulationSnapshot(current_minute, satellites, jobs)
    inspect_snapshot(snapshot, generation=0, reset_count=0, label="Initial simulation snapshot")
    last_evaluation_snapshot = None
    metrics_collector = None
    if collect_metrics:
        run_prefix = metrics_prefix.strip() or datetime.now().strftime("ga_%Y%m%d_%H%M%S")
        metrics_collector = MetricsCollector(metrics_dir, run_prefix, metrics_flush_every)

    best_overall_fitness = float('-inf')
    best_world_fitness = float('-inf')
    generations_without_improvement = 0
    total_resets = 0
    #endregion



    #region > Evolution loop
    for generation in range(generations):
        


        #region >> Evaluation
        # Evaluare fitness of population by scheduling tasks
        population = genetic_algorithm.getPopulation()
        fitness_evaluator.evaluate(population, snapshot)
        evaluation_snapshot = snapshot
        last_evaluation_snapshot = evaluation_snapshot
        #endregion



        #region >> Convergence
        current_best = max(population, key=lambda ind: ind.fitness)
        if current_best.fitness > best_overall_fitness:
            print(f"New best overall fitness: {current_best.fitness:.4f} (previous: {best_overall_fitness:.4f})")
            best_overall_fitness = current_best.fitness
        
        if current_best.fitness > best_world_fitness:
            print(f"New best fitness in world: {current_best.fitness:.4f} (previous: {best_world_fitness:.4f})")
            best_world_fitness = current_best.fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        world_reset = 0
        if generations_without_improvement >= convergence_threshold:
            # Regenerate the world
            print(f"Convergence detected after {generations_without_improvement} generations without improvement. Regenerating satellites and jobs...")
            reset_job_limit = initial_job_limit
            next_world_path = select_world_path(total_resets + 1)
            satellites, jobs, current_minute, world_seed = build_world(next_world_path, reset_job_limit)
            print(f"Loaded world '{next_world_path}' for reset (seed={world_seed}).")
            snapshot = SimulationSnapshot(current_minute, satellites, jobs)
            total_resets += 1
            inspect_snapshot(
                snapshot,
                generation=generation + 1,
                reset_count=total_resets,
                label=f"World reset #{total_resets} at generation {generation + 1}",
            )
            best_world_fitness = float('-inf')
            generations_without_improvement = 0
            world_reset = 1
        #endregion



        #region >> Metrics 
        if metrics_collector is not None:
            fitness_values = [individual.fitness for individual in population]
            mean_fitness = sum(fitness_values) / len(fitness_values)
            variance = sum((value - mean_fitness) ** 2 for value in fitness_values) / len(fitness_values)
            ratio, resp, energy = fitness_evaluator.evaluate_single(current_best, evaluation_snapshot)

            metrics_collector.record(
                GenerationMetrics(
                    generation=generation + 1,
                    best_fitness=current_best.fitness,
                    mean_fitness=mean_fitness,
                    fitness_std=variance ** 0.5,
                    best_completion_ratio=ratio,
                    best_mean_response=resp,
                    best_energy=energy,
                    jobs_in_snapshot=len(evaluation_snapshot.jobs),
                    satellites_in_snapshot=len(evaluation_snapshot.satellites),
                    stagnation_generations=generations_without_improvement,
                    world_reset=world_reset,
                )
            )
        #endregion



        #region >> Logs
        if print_every > 0 and ((generation + 1) % print_every == 0 or generation == generations - 1):
            print("SCHEDULE SNAPSHOT " + "-" * 50)
            genetic_algorithm.printSchedule()
            print("\n")
        #endregion



        #region >> Evolution
        # Evolve the population
        if generation < generations - 1:
            genetic_algorithm.solveNextGeneration()
        #endregion
    #endregion



    #region > Post-evolution analysis
    # After evolution: inspect the winner
    best = max(genetic_algorithm.getPopulation(), key=lambda ind: ind.fitness)
    ratio, resp, energy = fitness_evaluator.evaluate_single(best, last_evaluation_snapshot)
    print(f"Jobs scheduled: {ratio:.0%}  |  Mean response: {resp:.1f} min  |  Energy: {energy:.1f} Wmin")
    inspect_snapshot(
        last_evaluation_snapshot,
        generation=generations,
        reset_count=total_resets,
        label="Final evaluation snapshot",
    )

    if metrics_collector is not None:
        metrics_collector.close()
        print(f"Metrics CSV: {metrics_collector.csv_path}")
        print(f"Metrics NumPy: {metrics_collector.npz_path}")
        print(f"Metrics graph: {metrics_collector.plot_path}")
    #endregion
#endregion



#region Entry point
if __name__ == "__main__":
    import os, json
    print(f"PID: {os.getpid()} — attach GDB now if you want to debug the C++ code.")

    args = parser.parse_args()

    # --- Build config: defaults → JSON file (if -c given) → explicit CLI flags ---
    _known_keys = set(GAConfig.__annotations__)

    config = default_config()

    if args.config is not None:
        config_path = os.path.join("runConfigurations", f"{args.config}.json")
        with open(config_path) as f:
            json_data = json.load(f)
        for key, value in json_data.items():
            if key in _known_keys:
                config[key] = value   # type: ignore[literal-required]
        print(f"Loaded run configuration: {config_path}")

    # CLI flags that differ from their argparse defaults override the JSON.
    # We detect explicit overrides by comparing against argparse defaults.
    _cli_overrides: dict = {}
    _defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    for dest, default in _defaults.items():
        cli_value = getattr(args, dest, None)
        if cli_value != default:
            _cli_overrides[dest] = cli_value

    _dest_to_key = {
        "generations":          "generations",
        "population":           "population_size",
        "max_depth":            "max_depth",
        "hoist_rate":           "hoist_rate",
        "convergence_threshold": "convergence_threshold",
        "csv_row_limit":        "csv_row_limit",
        "metrics_dir":          "metrics_dir",
        "metrics_prefix":       "metrics_prefix",
        "metrics_flush_every":  "metrics_flush_every",
        "disable_metrics":      None,   # handled separately below
        "inspect_snapshots":    "inspect_snapshots",
        "save_snapshots":       "save_snapshots",
        "snapshots_dir":        "snapshots_dir",
        "snapshots_prefix":     "snapshots_prefix",
        "print_every":          "print_every",
    }
    for dest, key in _dest_to_key.items():
        if dest in _cli_overrides and key is not None:
            config[key] = _cli_overrides[dest]  # type: ignore[literal-required]
    if "disable_metrics" in _cli_overrides:
        config["collect_metrics"] = not _cli_overrides["disable_metrics"]
    # -------------------------------------------------------------------------

    if not args.no_start_prompt:
        print("Press Enter to start the genetic algorithm...")
        input()

    run_genetic_algorithm(config)
#endregion