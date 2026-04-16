
#include <cstddef>
#include <iostream>

#include "scheduler/data_factories.h"
#include "scheduler/scheduler_gp.h"

using namespace std;

int gp_main() {
    auto satellites = createSatellites();
    auto jobs       = createJobs();

    cout << "=== LEO Satellite Scheduler — Genetic Programming ===\n";
    cout << "Terminals: ";
    auto reg = buildTerminalRegistry();
    for (size_t i = 0; i < reg.size(); ++i)
        cout << reg[i].name << (i + 1 < reg.size() ? ", " : "\n");
    cout << "Operators: +, -, *, /safe, MAX, MIN, NEG, ABS, INV\n\n";

    SchedulerGP gp(satellites, jobs);

    GPIndividual best = gp.solve(
        /*populationSize=*/ 200,
        /*generations=*/    300,
        /*crossoverRate=*/  0.80,
        /*mutationRate=*/   0.08
    );

    gp.printSchedule(best);
    return 0;
}