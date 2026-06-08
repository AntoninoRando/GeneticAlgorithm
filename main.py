import argparse
import random
from datetime import datetime

from scheduler import SchedulerGP, FitnessEvaluator, SimulationSnapshot, buildTerminalRegistry
from data_factory import create_jobs_from_csv, create_satellites_from_csv
from metrics_tracker import GenerationMetrics, MetricsCollector
from snapshot_inspector import print_simulation_snapshot_summary, save_simulation_snapshot



#region ARGS
parser = argparse.ArgumentParser(description="Run GP Scheduler")


#region GA parameters
parser.add_argument("--population", type=int, default=100, help="Population size")
parser.add_argument("--generations", type=int, default=200, help="Number of generations")
parser.add_argument("--print-every", type=int, default=50, help="Print progress every N generations")
parser.add_argument("--max-depth", type=int, default=7,
                    help="Maximum expression tree depth. Shallower trees are faster and simpler "
                         "(e.g. --max-depth 4). Max nodes ≈ 2^(depth+1), so depth 7 → ≤256 nodes.")
parser.add_argument("--hoist-rate", type=float, default=0.05,
                    help="Probability [0..1] that each new offspring undergoes a hoist mutation, "
                         "which replaces the whole tree with one of its own subtrees to fight bloat "
                         "(e.g. --hoist-rate 0.20 for aggressive pruning).")
parser.add_argument("--convergence-threshold", type=int, default=10, 
                    help="Number of generations without fitness improvement before regenerating satellites and jobs (G).")
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
#endregion
#endregion



def run_genetic_algorithm(generations: int, population_size: int, print_every: int,
                          max_depth: int, hoist_rate: float, convergence_threshold: int = 10,
                          metrics_dir: str = "metrics", metrics_prefix: str = "",
                          metrics_flush_every: int = 1, collect_metrics: bool = True,
                          inspect_snapshots: bool = False, save_snapshots: bool = False,
                          snapshots_dir: str = "snapshots", snapshots_prefix: str = ""):
    if generations <= 0:
        print("No generations requested; nothing to optimize.")
        return

    csv_path = "data/example.csv"
    initial_job_limit = 300
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

    def build_world(job_limit: int, minute: float):
        world_seed = world_rng.randrange(0, 2_147_483_647)
        satellites_local = create_satellites_from_csv(
            csv_path,
            limit=job_limit,
            seed=world_seed,
            random_sample=False,
            sample_seed=world_seed,
        )
        jobs_local = create_jobs_from_csv(
            csv_path,
            limit=job_limit,
            snapshot_minute=minute,
            random_sample=False,
            sample_seed=world_seed,
        )
        return satellites_local, jobs_local, world_seed

    satellites, jobs, _ = build_world(initial_job_limit, current_minute)

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
    generations_without_improvement = 0
    total_resets = 0
    
    for generation in range(generations):
        # Evaluare fitness of population by scheduling tasks
        population = genetic_algorithm.getPopulation()
        fitness_evaluator.evaluate(population, snapshot)
        evaluation_snapshot = snapshot
        last_evaluation_snapshot = evaluation_snapshot
        
        # Check convergence
        current_best = max(population, key=lambda ind: ind.fitness)
        if current_best.fitness > best_overall_fitness:
            best_overall_fitness = current_best.fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        world_reset = 0
        if generations_without_improvement >= convergence_threshold:
            # Regenerate the world
            print(f"Convergence detected after {generations_without_improvement} generations without improvement. Regenerating satellites and jobs...")
            reset_job_limit = 300
            current_minute = generation * 5.0
            satellites, jobs, world_seed = build_world(reset_job_limit, current_minute)
            print(f"Loaded new CSV sample for reset (seed={world_seed}).")
            snapshot = SimulationSnapshot(current_minute, satellites, jobs)
            total_resets += 1
            inspect_snapshot(
                snapshot,
                generation=generation + 1,
                reset_count=total_resets,
                label=f"World reset #{total_resets} at generation {generation + 1}",
            )
            best_overall_fitness = float('-inf')
            generations_without_improvement = 0
            world_reset = 1

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

        # Print progress
        if print_every > 0 and ((generation + 1) % print_every == 0 or generation == generations - 1):
            print("SCHEDULE SNAPSHOT " + "-" * 50)
            genetic_algorithm.printSchedule()
            print("\n")

        # Evolve the population
        if generation < generations - 1:
            genetic_algorithm.solveNextGeneration()

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



if __name__ == "__main__":
    import os
    print(f"PID: {os.getpid()} — attach GDB now if you want to debug the C++ code.")

    args = parser.parse_args()

    if not args.no_start_prompt:
        print("Press Enter to start the genetic algorithm...")
        input()

    run_genetic_algorithm(args.generations, args.population, args.print_every,
                          args.max_depth, args.hoist_rate, args.convergence_threshold,
                          args.metrics_dir, args.metrics_prefix, args.metrics_flush_every,
                          not args.disable_metrics, args.inspect_snapshots, args.save_snapshots,
                          args.snapshots_dir, args.snapshots_prefix)
