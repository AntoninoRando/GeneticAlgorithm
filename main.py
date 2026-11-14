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
    # Opt-in refinements from the GP-for-scheduling literature (see the
    # LITERATURE NOTES in include/scheduler/*.h).  All default to the earlier
    # behaviour, so existing runs and the thesis experiments are unchanged.
    tournament_size: int
    initial_max_depth: int
    internal_crossover_bias: float
    clone_retries: int
    depth_aware_mutation: bool
    depth_aware_crossover: bool
    terminal_weighting: bool
    track_busy_until: bool
    fixed_bounds: bool
    earliest_free_tie_break: bool
    prune_constant_terminals: bool
    worlds_per_eval: int
    # Reproducibility: -1 = fresh randomness every run; >= 0 = deterministic
    # run (seeds both the C++ GP evolution and the Python world sampling).
    seed: int
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
        tournament_size=5,
        initial_max_depth=-1,
        internal_crossover_bias=0.0,
        clone_retries=0,
        depth_aware_mutation=False,
        depth_aware_crossover=False,
        terminal_weighting=False,
        track_busy_until=False,
        fixed_bounds=False,
        earliest_free_tie_break=False,
        prune_constant_terminals=False,
        worlds_per_eval=1,
        seed=-1,
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
# No option carries an argparse default (argument_default=SUPPRESS): the parsed
# namespace then contains ONLY the options that were typed on the command line.
# That is what lets resolve_config() tell "--generations 200 was typed" from
# "--generations was left out", so a typed flag always beats the JSON file,
# even when its value equals the built-in default.  The built-in defaults live
# in one place, default_config().  Every option's dest is its GAConfig key.
parser = argparse.ArgumentParser(
    description="Run GP Scheduler",
    argument_default=argparse.SUPPRESS,
    epilog="Built-in defaults (overridden by -c FILE, then by the flags you type): "
           + ", ".join(f"{key}={value!r}" for key, value in default_config().items()),
)


#region GA parameters
parser.add_argument("--population", type=int, dest="population_size", help="Population size")
parser.add_argument("--generations", type=int, help="Number of generations")
parser.add_argument("--max-depth", type=int,
                    help="Maximum expression tree depth. Shallower trees are faster and simpler "
                         "(e.g. --max-depth 4). Max nodes ≈ 2^(depth+1), so depth 7 → ≤256 nodes.")
parser.add_argument("--hoist-rate", type=float,
                    help="Probability [0..1] that each new offspring undergoes a hoist mutation, "
                         "which replaces the whole tree with one of its own subtrees to fight bloat "
                         "(e.g. --hoist-rate 0.20 for aggressive pruning).")
parser.add_argument("--convergence-threshold", type=int, 
                    help="Number of generations without fitness improvement before regenerating satellites and jobs (G).")
parser.add_argument("--csv-row-limit", type=int,
                    help="Maximum number of rows to load from the CSV file for satellites and jobs. "
                         "Set to 0 or a negative number to load all rows (e.g. --csv-row-limit 50).")
parser.add_argument("--tournament-size", type=int,
                    help="Candidates per selection tournament (selection pressure). The papers use "
                         "7 (Sun 2024, Yang 2025) or 2 (Zhang & Yang 2025).")
parser.add_argument("--initial-max-depth", type=int,
                    help="Maximum depth of the initial random trees; -1 = same as --max-depth. "
                         "The papers start below the cap (e.g. 2-6 with a cap of 8 in Yang 2025) so "
                         "that crossover has room to grow trees.")
parser.add_argument("--internal-crossover-bias", type=float,
                    help="Probability [0..1] that a crossover point is an operator node rather than "
                         "a leaf. 0 = uniform over all nodes; Koza (1994) uses 0.9.")
parser.add_argument("--clone-retries", type=int,
                    help="Re-mutation attempts for an offspring identical to a parent or an elite "
                         "(structural proxy of the repetition test in Zhang & Yang 2025). 0 = off.")
parser.add_argument("--depth-aware-mutation", action=argparse.BooleanOptionalAction,
                    help="Subtree mutation grows the replacement within the depth left below the cap "
                         "instead of always at depth 3, so a mutation is never discarded for exceeding "
                         "--max-depth (Poli, Langdon & McPhee 2008, Sec. 11.3.2).")
parser.add_argument("--depth-aware-crossover", action=argparse.BooleanOptionalAction,
                    help="Draw the second crossover point among the nodes for which both children "
                         "respect --max-depth, so crossover never falls back to a copy of a parent "
                         "(Poli, Langdon & McPhee 2008, Sec. 11.3.2; Langdon 2000).")
parser.add_argument("--terminal-weighting", action=argparse.BooleanOptionalAction,
                    help="Adaptive terminal weighting for mutation: terminal probabilities follow the "
                         "maximum-likelihood fit to the best individuals (Yang et al. 2025, Sec. IV-D).")
parser.add_argument("--track-busy-until", action=argparse.BooleanOptionalAction,
                    help="Make the sat.cpuBusyUntil terminal live: the decoder writes each satellite's "
                         "next free minute into it (availability/queue-load feature of Zhang & Yang "
                         "2025 and Sun et al. 2024).")
parser.add_argument("--fixed-bounds", action=argparse.BooleanOptionalAction,
                    help="Fixed-bound fitness: divide response time and energy by per-world bounds "
                         "instead of the population's min and max, and charge every unscheduled job "
                         "the worst response (Marler & Arora 2004; Shabtay et al. 2013). The fitness "
                         "becomes absolute, so it is comparable across generations.")
parser.add_argument("--earliest-free-tie-break", action=argparse.BooleanOptionalAction,
                    help="Decoder tie-break: among equally scored (job, satellite) pairs prefer the "
                         "satellite that is free earliest, instead of the lowest index. A rule that "
                         "ignores the satellite then spreads its jobs like list scheduling (Graham "
                         "1966) rather than queueing them on the first satellite.")
parser.add_argument("--prune-constant-terminals", action=argparse.BooleanOptionalAction,
                    help="Never draw terminals that are constant over every loaded world and not "
                         "updated by the decoder (e.g. sat.energyReserved, sat.loadRatio).")
parser.add_argument("--worlds-per-eval", type=int,
                    help="Number of worlds each individual is scored on per generation (pooled). "
                         "The papers score every individual on several instances (5 in Yang 2025, "
                         "100 simulations in Zhang & Yang 2025). 1 = a single world, as before.")
parser.add_argument("--seed", type=int,
                    help="Random seed for reproducible runs (seeds both the C++ evolution and the "
                         "Python world sampling). -1 (default) uses fresh randomness every run.")
#endregion



#region Metrics
parser.add_argument("--metrics-dir", type=str,
                    help="Directory where metrics artifacts are written.")
parser.add_argument("--metrics-prefix", type=str,
                    help="Output prefix for metrics files. Defaults to a timestamp.")
parser.add_argument("--metrics-flush-every", type=int,
                    help="How often (in generations) to refresh the NumPy metrics dump.")
parser.add_argument("--disable-metrics", action="store_false", dest="collect_metrics",
                    help="Disable metrics collection and graph generation.")
parser.add_argument("--enable-metrics", action="store_true", dest="collect_metrics",
                    help="Enable metrics collection even if the run configuration turns it off "
                         "(runConfigurations/default.json does).")
#endregion



#region Snapshots
parser.add_argument("--inspect-snapshots", action=argparse.BooleanOptionalAction,
                    help="Print readable summaries for initial/reset/final simulation snapshots.")
parser.add_argument("--save-snapshots", action=argparse.BooleanOptionalAction,
                    help="Persist inspected snapshots as JSON files.")
parser.add_argument("--snapshots-dir", type=str,
                    help="Directory where snapshot JSON files are written.")
parser.add_argument("--snapshots-prefix", type=str,
                    help="Output prefix for snapshot JSON files. Defaults to a timestamp.")
#endregion


#region Miscellaneous
parser.add_argument("--no-start-prompt", action="store_true",
                    help="Start immediately without waiting for Enter.")
parser.add_argument("--print-every", type=int, help="Print progress every N generations")
#endregion


#region Config file
parser.add_argument("-c", "--config", type=str,
                    help="Name of a run configuration (without extension) in the runConfigurations/ "
                         "directory, e.g. '--config default' loads runConfigurations/default.json. "
                         "Known keys override the built-in defaults; unrecognized keys are ignored. "
                         "Explicit CLI flags always take precedence over the JSON values.")
#endregion
#endregion



#region Configuration layering
def resolve_config(argv=None, config_dir: str = "runConfigurations"):
    """Builds the run configuration: built-in defaults, then the JSON file
    named by -c/--config, then the flags typed on the command line.

    Returns ``(config, start_prompt)``; ``start_prompt`` is False when
    --no-start-prompt was given.

    A flag overrides the file whenever it was TYPED, whatever its value.
    (The previous version compared each value with its argparse default and so
    ignored a typed flag that happened to equal the default: with
    ``-c default --generations 200`` the file's 5000 won.)  Boolean options
    come in pairs (``--fixed-bounds`` / ``--no-fixed-bounds``,
    ``--enable-metrics`` / ``--disable-metrics``) so that a value the file sets
    to true can also be switched off from the command line.

    ``argv`` and ``config_dir`` exist for the tests; main() uses sys.argv and
    the runConfigurations/ directory.
    """
    import json
    import os

    typed = vars(parser.parse_args(argv))      # only what was typed, see the parser
    known_keys = set(GAConfig.__annotations__)

    config = default_config()

    config_name = typed.get("config")
    if config_name is not None:
        config_path = os.path.join(config_dir, f"{config_name}.json")
        with open(config_path) as f:
            json_data = json.load(f)
        for key, value in json_data.items():
            if key in known_keys:
                config[key] = value   # type: ignore[literal-required]
        print(f"Loaded run configuration: {config_path}")

    for key, value in typed.items():
        if key in known_keys:
            config[key] = value       # type: ignore[literal-required]

    return config, not typed.get("no_start_prompt", False)
#endregion



#region Main loop
def run_genetic_algorithm(config: GAConfig):
    generations        = config["generations"]
    population_size    = config["population_size"]
    max_depth          = config["max_depth"]
    hoist_rate         = config["hoist_rate"]
    convergence_threshold = config["convergence_threshold"]
    csv_row_limit      = config["csv_row_limit"]
    tournament_size    = config["tournament_size"]
    initial_max_depth  = config["initial_max_depth"]
    internal_crossover_bias = config["internal_crossover_bias"]
    clone_retries      = config["clone_retries"]
    depth_aware_mutation  = config["depth_aware_mutation"]
    depth_aware_crossover = config["depth_aware_crossover"]
    terminal_weighting = config["terminal_weighting"]
    track_busy_until   = config["track_busy_until"]
    fixed_bounds       = config["fixed_bounds"]
    earliest_free_tie_break = config["earliest_free_tie_break"]
    prune_constant_terminals = config["prune_constant_terminals"]
    worlds_per_eval    = max(1, config["worlds_per_eval"])
    seed               = config["seed"]
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
        # data/ is not in the repository (.gitignore), so on a fresh clone
        # neither exists.  Say so, instead of failing later with a bare
        # FileNotFoundError from the CSV loader.
        from pathlib import Path
        if not (Path(__file__).resolve().parent / world_paths[0]).is_file():
            raise SystemExit(
                "No data found: neither data/worlds/ nor data/example.csv exists.\n"
                "The simulation trace is not stored in the repository. Copy it to "
                "data/example.csv, then run\n"
                "    python3 -m data_processing.csv_preprocessing.world_splitter\n"
                "to create data/worlds/ (see the Data section of readme.md).")
        print("No pre-processed worlds in data/worlds/; falling back to "
              "data/example.csv. Run "
              "`python -m data_processing.csv_preprocessing.world_splitter` "
              "to generate worlds.")

    def select_world_path(world_index: int) -> str:
        return world_paths[world_index % len(world_paths)]

    initial_job_limit = csv_row_limit
    current_minute = 0.0
    world_rng = random.Random(seed if seed >= 0 else None)
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

    def build_snapshots(first_world_index: int):
        """Builds the `worlds_per_eval` snapshots that the population is scored on.

        Papers never score an individual on a single instance: Yang et al. 2025
        draw 5 training instances per individual per generation (Sec. V-A,
        thesis Sec. 3.3.1) and Zhang & Yang 2025 average 100 simulations per
        individual (Sec. 4.1).  With one world the GP can overfit that world's
        quirks (the convergence-triggered rotation below only limits this
        across time); several worlds at once do it within a generation.
        Consecutive worlds are used so that a reset moves to fresh ones.
        """
        built = []
        for k in range(worlds_per_eval):
            sats_k, jobs_k, minute_k, _ = build_world(
                select_world_path(first_world_index + k), initial_job_limit)
            built.append((sats_k, jobs_k, minute_k, SimulationSnapshot(minute_k, sats_k, jobs_k)))
        return built

    world_pool = build_snapshots(0)
    satellites, jobs, current_minute, snapshot = world_pool[0]
    snapshots = [entry[3] for entry in world_pool]

    # The evaluator is built first because it also tells the GP which terminals
    # are dead (constant in every world), see include/scheduler/registry.h.
    fitness_evaluator = FitnessEvaluator(buildTerminalRegistry())
    if track_busy_until:
        fitness_evaluator.trackBusyUntil = True
    if fixed_bounds:
        # Opt-in, see the "Fixed-bound normalisation" note in
        # include/scheduler/fitness_evaluator.h.  Set only when requested, so a
        # `scheduler` binary built before the option existed still runs the
        # default mode (it raises AttributeError only if this is asked for).
        fitness_evaluator.fixedBounds = True
    if earliest_free_tie_break:
        # Opt-in decoder tie-break, see FitnessEvaluator::earliestFreeTieBreak.
        # Set only when requested, for the same reason as above.
        fitness_evaluator.earliestFreeTieBreak = True

    excluded_terminals = []
    if prune_constant_terminals:
        # A terminal is excluded only if it is constant in *all* loaded worlds
        # (intersection): one that varies somewhere may still carry signal.
        per_world = [set(fitness_evaluator.constant_terminals(sn)) for sn in snapshots]
        excluded_terminals = sorted(set.intersection(*per_world))
        names = [t.name for t in buildTerminalRegistry()]
        print("Excluding constant terminals: "
              + (", ".join(names[i] for i in excluded_terminals) or "none"))

    # Only options that differ from their defaults are passed on, so a
    # `scheduler` binary built before these options existed keeps working in
    # the default mode (it raises a clear TypeError only if one is requested).
    gp_options = {}
    if tournament_size != 5:           gp_options["tournamentSize"] = tournament_size
    if initial_max_depth >= 0:         gp_options["initialMaxDepth"] = initial_max_depth
    if internal_crossover_bias > 0:    gp_options["internalCrossoverBias"] = internal_crossover_bias
    if clone_retries > 0:              gp_options["cloneRetries"] = clone_retries
    if depth_aware_mutation:           gp_options["depthAwareMutation"] = True
    if depth_aware_crossover:          gp_options["depthAwareCrossover"] = True
    if terminal_weighting:             gp_options["terminalWeighting"] = True
    if excluded_terminals:             gp_options["excludedTerminals"] = excluded_terminals

    genetic_algorithm = SchedulerGP(satellites, jobs, seed=seed)
    genetic_algorithm.initialize(populationSize=population_size, maxDepth=max_depth,
                                 hoistRate=hoist_rate, **gp_options)
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
        if len(snapshots) == 1:
            fitness_evaluator.evaluate(population, snapshot)
        else:
            fitness_evaluator.evaluate_many(population, snapshots)
        # Raw per-generation metrics below are reported on the first world only.
        evaluation_snapshot = snapshot
        last_evaluation_snapshot = evaluation_snapshot
        #endregion



        #region >> Convergence
        # NOTE: fitness is population-relative (penalties are normalised across
        # the current population) and world-dependent, so best_overall_fitness
        # is only an informational indicator across world resets; the actual
        # convergence decision uses best_world_fitness, which resets per world.
        # With --fixed-bounds the fitness is absolute within a world set, so
        # "no new best for G generations" is then an exact stagnation test.
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
            # Regenerate the world.  Stagnation-triggered rotation is this
            # project's version of the "rotation" instance-sampling strategy
            # that Yang (2025, thesis Sec. 3.6) calls the default in GPHH; the
            # thesis devotes Chapter 4 to comparing it with other strategies.
            print(f"Convergence detected after {generations_without_improvement} generations without improvement. Regenerating satellites and jobs...")
            first_next = (total_resets + 1) * worlds_per_eval
            world_pool = build_snapshots(first_next)
            satellites, jobs, current_minute, snapshot = world_pool[0]
            snapshots = [entry[3] for entry in world_pool]
            print(f"Loaded {len(world_pool)} world(s) starting at "
                  f"'{select_world_path(first_next)}' for reset.")
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
            if hasattr(genetic_algorithm, "structuralDiversity"):
                print(f"Structural diversity: {genetic_algorithm.structuralDiversity():.2f}")
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
    import os
    print(f"PID: {os.getpid()} — attach GDB now if you want to debug the C++ code.")

    # defaults → JSON file (if -c given) → flags typed on the command line
    config, start_prompt = resolve_config()

    if start_prompt:
        print("Press Enter to start the genetic algorithm...")
        input()

    run_genetic_algorithm(config)
#endregion