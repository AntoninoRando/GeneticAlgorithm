import scheduler  # the compiled .so module
import random

def create_satellites():
    satellites = [
        scheduler.Satellite(0, "SEO-A"),
        scheduler.Satellite(1, "SEO-B"),
        scheduler.Satellite(2, "SEO-C"),
    ]

    satellites[0].listeningDome = [scheduler.Satellite(1, "SEO-B"), scheduler.Satellite(2, "SEO-C")]
    satellites[1].listeningDome = [scheduler.Satellite(0, "SEO-A"), scheduler.Satellite(2, "SEO-C")]
    satellites[2].listeningDome = [scheduler.Satellite(0, "SEO-A"), scheduler.Satellite(1, "SEO-B")]

    satellites[0].activeTasks = 0
    satellites[1].activeTasks = 0
    satellites[2].activeTasks = 0

    satellites[0].RemainingEnergy = 100.0
    satellites[1].RemainingEnergy = 98.0
    satellites[2].RemainingEnergy = 95.0

    satellites[0].ComputingLoad = 0.20
    satellites[1].ComputingLoad = 0.25
    satellites[2].ComputingLoad = 0.18

    satellites[0].ComputingCapability = 18.0
    satellites[1].ComputingCapability = 16.5
    satellites[2].ComputingCapability = 19.0

    return satellites

def create_jobs():
    mission_start_epoch = 1700000000

    jobs = [
        scheduler.Job(0, "Target-01", 8, 100),
        scheduler.Job(1, "Target-02", 10, 145),
        scheduler.Job(2, "Target-03", 12, 190),
        scheduler.Job(3, "Target-04", 9, 205),
        scheduler.Job(4, "Target-05", 7, 230),
        scheduler.Job(5, "Target-06", 11, 280),
        scheduler.Job(6, "Target-07", 8, 315),
        scheduler.Job(7, "Target-08", 10, 350),
        scheduler.Job(8, "Target-09", 12, 390),
        scheduler.Job(9, "Target-10", 9, 430),
    ]

    arrivals = [10, 55, 105, 138, 165, 205, 245, 285, 320, 350]
    task_sizes = [10.0, 14.0, 17.0, 9.5, 11.0, 16.0, 10.5, 13.5, 18.0, 12.0]

    for i, job in enumerate(jobs):
        job.arrivalTime = mission_start_epoch + arrivals[i] * 60
        job.taskSize = task_sizes[i]
        job.initialDeadline = float(job.dueMinute)
        job.remainingDeadline = float(job.dueMinute)

    return jobs

def run_genetic_algorithm(population_size=160, generations=500,
                           crossover_rate=0.85, mutation_rate=0.06,
                           elitism_count=2, print_every=25):
    satellites = create_satellites()
    jobs = create_jobs()

    ga = scheduler.SchedulerGP(satellites, jobs)
    ga.initialize(population_size, crossover_rate, mutation_rate, elitism_count)

    for generation_index in range(generations):
        # ! IMPORTANT
        # Run the actual simulation. For now, randomly assign fitness values.
        population = ga.getPopulation()
        for individual in population:
            # Simulate the individual's tree and set its fitness based on the results.
            # Here we use a random fitness for demonstration purposes.
            individual.fitness = random.uniform(0, 100)

        ga.solveNextGeneration()

        if (generation_index + 1) % print_every == 0 or generation_index == generations - 1:
            ga.printSchedule()
            #print_best_individual(generation_index, ga.getPopulation())

    population = ga.getPopulation()

if __name__ == "__main__":
    run_genetic_algorithm()