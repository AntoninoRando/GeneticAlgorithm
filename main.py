import scheduler  # the compiled .so module

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
    def opp(sat, start, end):
        return scheduler.Opportunity(sat, start, end)

    mission_start_epoch = 1700000000

    jobs = [
        scheduler.Job(0, "Target-01", 8, 100, 5.0, [opp(0, 30, 38), opp(1, 42, 50), opp(2, 55, 63)]),
        scheduler.Job(1, "Target-02", 10, 145, 7.0, [opp(0, 70, 80), opp(1, 92, 102), opp(2, 110, 120)]),
        scheduler.Job(2, "Target-03", 12, 190, 9.0, [opp(0, 125, 137), opp(1, 140, 152), opp(2, 162, 174)]),
        scheduler.Job(3, "Target-04", 9, 205, 4.0, [opp(0, 155, 164), opp(1, 175, 184), opp(2, 188, 197)]),
        scheduler.Job(4, "Target-05", 7, 230, 6.5, [opp(0, 180, 187), opp(1, 198, 205), opp(2, 216, 223)]),
        scheduler.Job(5, "Target-06", 11, 280, 8.5, [opp(0, 220, 231), opp(1, 240, 251), opp(2, 255, 266)]),
        scheduler.Job(6, "Target-07", 8, 315, 5.5, [opp(0, 260, 268), opp(1, 272, 280), opp(2, 295, 303)]),
        scheduler.Job(7, "Target-08", 10, 350, 7.5, [opp(0, 300, 310), opp(1, 320, 330), opp(2, 338, 348)]),
        scheduler.Job(8, "Target-09", 12, 390, 9.5, [opp(0, 335, 347), opp(1, 355, 367), opp(2, 370, 382)]),
        scheduler.Job(9, "Target-10", 9, 430, 6.0, [opp(0, 365, 374), opp(1, 385, 394), opp(2, 405, 414)]),
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
                           crossover_rate=0.85, mutation_rate=0.06):
    satellites = create_satellites()
    jobs = create_jobs()

    ga = scheduler.SchedulerGA(satellites, jobs)
    best = ga.solve(population_size, generations, crossover_rate, mutation_rate)

    print(f"Fitness:        {best.fitness:.2f}")
    print(f"Scheduled jobs: {best.scheduledJobs}")
    print(f"Conflicts:      {best.conflicts}")
    print(f"Lateness:       {best.lateness} min")

    ga.printSchedule(best)
    return best

if __name__ == "__main__":
    run_genetic_algorithm()