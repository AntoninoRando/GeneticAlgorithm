import argparse
from scheduler import SchedulerGP, FitnessEvaluator, SimulationSnapshot, buildTerminalRegistry
from data_factory import create_satellites, create_jobs



parser = argparse.ArgumentParser(description="Run GP Scheduler")
parser.add_argument("--population", type=int, default=100, help="Population size")
parser.add_argument("--generations", type=int, default=200, help="Number of generations")
parser.add_argument("--print-every", type=int, default=-1, help="Print progress every N generations")
args = parser.parse_args()



def run_genetic_algorithm(generations: int, population_size: int, print_every: int):
    satellites = create_satellites()
    jobs = create_jobs()

    genetic_algorithm   = SchedulerGP(satellites, jobs)
    genetic_algorithm.initialize(populationSize=population_size)
    fitness_evaluator = FitnessEvaluator(buildTerminalRegistry())
    snapshot = None

    for generation in range(generations):
        # Regenerate the world every iteration
        satellites = create_satellites(n=10)
        jobs       = create_jobs(n=100, start_minute=generation * 5.0)
        current_minute = generation * 5.0
        snapshot = SimulationSnapshot(current_minute, satellites, jobs)

        # Evaluare fitness of population by scheduling tasks
        fitness_evaluator.evaluate(genetic_algorithm.getPopulation(), snapshot)

        # Evolve the population
        genetic_algorithm.solveNextGeneration()

        # Print progress
        if print_every <= 0:
            continue
        if (generation + 1) % print_every == 0 or generation == generations - 1:
            genetic_algorithm.printSchedule()

    # After evolution: inspect the winner
    best = max(genetic_algorithm.getPopulation(), key=lambda ind: ind.fitness)
    ratio, resp, energy = fitness_evaluator.evaluate_single(best, snapshot)
    print(f"Jobs scheduled: {ratio:.0%}  |  Mean response: {resp:.1f} min  |  Energy: {energy:.1f} Wmin")



if __name__ == "__main__":
    run_genetic_algorithm(args.generations, args.population, args.print_every)