from scheduler import SchedulerGP, FitnessEvaluator, SimulationSnapshot

gp   = SchedulerGP(satellites, jobs)
gp.initialize(populationSize=100)
eval = FitnessEvaluator(buildTerminalRegistry())

for generation in range(50):
    snap = SimulationSnapshot(current_minute, satellites, jobs)
    eval.evaluate(gp.getPopulation(), snap)   # writes fitness in place
    gp.solveNextGeneration()

# After evolution: inspect the winner
best = max(gp.getPopulation(), key=lambda ind: ind.fitness)
ratio, resp, energy = eval.evaluate_single(best, snap)
print(f"Jobs scheduled: {ratio:.0%}  |  Mean response: {resp:.1f} min  |  Energy: {energy:.1f} Wmin")