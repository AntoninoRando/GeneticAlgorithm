#pragma once

#include <vector>
#include <limits>
#include <string>
#include "types.h"

using namespace std;
// -----------------------------------------------------------------------------



/*
    This file defines the terminal registry for the genetic programming algorithm.
    The registry maps human-readable names to field accessors that can be used to
    evaluate nodes in the expression tree.

    Examples of terminals include:
        - "job.durationMinutes" to access the duration of a job.
        - "sat.RemainingEnergy" to access the remaining energy of a satellite.
        - "job.slack" to access the slack time of a job (derived from dueMinute 
          and duration).
        - "sat.loadRatio" to access the load ratio of a satellite (derived from 
          ComputingLoad and ComputingCapability).

    LITERATURE NOTES  (keys refer to thesis/references.bib; section numbers are
    those of the cited papers, see thesis/literature_notes.md)
    ----------------------------------------------------------------------
    How this terminal set compares with the ones evolved in the papers:

      this project             nearest terminal in the papers
      ---------------------    ------------------------------------------------
      job.arrivalTime          AT  (zhang2025mtgper Table 2); FCFS is this alone
      job.executionTime        ET  (mtgp_multicloud Table 2, no. 1)
      job.transferTime         VCT (mtgp_multicloud Table 2, no. 6), TS-driven
                               transmission delay (zhang2025mtgper §2.1.2)
      job.imageSize            TS  (zhang2025mtgper Table 2, task size)
      job.ram, job.disk        resource demands m_r, v_r of the feasibility
                               constraints (magliarisi2025orbit §4, Eqs. 8-9)
      sat.remainingEnergy      RE  (zhang2025mtgper Table 2)
      sat.cpuCapacity          CC  (zhang2025mtgper Table 2)
      sat.completedTasks       NT, number of tasks at the satellite
                               (zhang2025mtgper Table 2); in the decoder it is
                               the number assigned so far, i.e. a live queue
                               length
      sat.cpuBusyUntil         IAT/AAT, available time (mtgp_multicloud
                               Table 2, nos. 11 and 14), and the buffer load CL
                               (zhang2025mtgper Table 2).  Live only when
                               FitnessEvaluator::trackBusyUntil is on.

    Why the queue/availability terminals matter:  in zhang2025mtgper's feature
    analysis NT and CL are the terminals used most often by the evolved routing
    rule while RE and CC are rare (§5.2.2); in mtgp_multicloud the instance
    rule is dominated by EC, AAT and ST (§5.4).  The same quantity is the
    waiting time W of the response-time model of magliarisi2025orbit (§3.5,
    Eqs. 3-5).  A GP scheduler that cannot see how busy a satellite already is
    can only balance load through `sat.completedTasks`.

    Dead terminals.  Measured on the shipped worlds (data/worlds, 3 worlds
    inspected), `sat.cpuBusyUntil`, `sat.networkBusyUntil`, `sat.energyReserved`
    and `sat.loadRatio` are 0 for every satellite at the snapshot: the data
    factory never sets a backlog, reserved energy or a task list.  That is 4 of
    the 17 terminals (about 24% of random terminal draws) contributing a
    constant.  `sat.completedTasks` is also 0 at the snapshot but it is
    updated by the decoder, so it is NOT dead.  SchedulerGP can exclude dead
    terminals (see FitnessEvaluator::constantTerminals), which mirrors how the
    papers hand-pick a small, informative terminal set (8 in zhang2025mtgper,
    10 in gp_fog §VI-B, 22 in mtgp_multicloud §4.2).

    Measured: dropping the constant terminals (--prune-constant-terminals) and
    making cpuBusyUntil live (--track-busy-until) did NOT significantly change
    the held-out completion ratio on this data (0.221 and 0.232 against 0.223
    for the baseline, 20 seeds each, p = 0.47 and 0.33; thesis/experiments/
    run_ablation.py).  With only 5-30 minute jobs and a 100-job snapshot, a
    queue-aware rule has little to gain over a good static one.  Expect the
    availability terminal to matter more with deadlines and heavier loads, as
    in the papers.

    Not available: deadline terminals.  Jobs carry no deadline, so there is no
    equivalent of IDDL/RDDL (zhang2025mtgper), SD/ST (mtgp_multicloud) or the
    task deadline T_r^max of magliarisi2025orbit Eq. 7.  This is the main gap
    between this project's problem and those of the surveyed papers.
*/



#pragma region TYPES -----------------------------------------------------------
/// Plain function pointer, not std::function: terminal accessors are called in
/// the innermost loop of the greedy decoder (once per tree terminal per
/// (job, satellite) pair), and std::function's type-erased dispatch blocked
/// inlining there. All registry entries are captureless lambdas, which convert
/// to function pointers implicitly.
using FieldAccessor = double (*)(const Job&, const Satellite&);

struct TerminalDef {
    string        name;
    FieldAccessor accessor;
};
#pragma endregion --------------------------------------------------------------



#pragma region CORE ------------------------------------------------------------
/// @brief Builds the terminal registry, i.e. a mapping from human-readable 
/// names to field accessors. The field accessors are lambdas that take a Job 
/// and Satellite and return a double.
/// @return A vector of terminals.
static vector<TerminalDef> buildTerminalRegistry() {
    vector<TerminalDef> reg = {
    // ── Job fields ────────────────────────────────────────────────────────
        { "job.arrivalTime",       [](const Job& j, const Satellite&) { return static_cast<double>(j.arrivalTime);       } },
        { "job.ram",               [](const Job& j, const Satellite&) { return j.ram;                                    } },
        { "job.disk",              [](const Job& j, const Satellite&) { return j.disk;                                   } },
        { "job.imageSize",         [](const Job& j, const Satellite&) { return j.imageSize;                              } },
        { "job.executionTime",     [](const Job& j, const Satellite&) { return static_cast<double>(j.executionTime);     } },
        { "job.transferTime",      [](const Job& j, const Satellite&) { return static_cast<double>(j.transferTime);      } },
        { "job.numberOfHops",      [](const Job& j, const Satellite&) { return static_cast<double>(j.numberOfHops);      } },
    // ── Satellite fields ──────────────────────────────────────────────────
        { "sat.elevationAngle",    [](const Job&, const Satellite& s) { return s.elevationAngle;                         } },
        { "sat.latency",           [](const Job&, const Satellite& s) { return s.latency;                                } },
        { "sat.bandwidth",         [](const Job&, const Satellite& s) { return s.bandwidth;                              } },
        { "sat.completedTasks",    [](const Job&, const Satellite& s) { return static_cast<double>(s.completedTasks);    } },
        { "sat.cpuBusyUntil",      [](const Job&, const Satellite& s) { return static_cast<double>(s.cpuBusyUntil);      } },
        { "sat.networkBusyUntil",  [](const Job&, const Satellite& s) { return static_cast<double>(s.networkBusyUntil);  } },
        { "sat.cpuCapacity",       [](const Job&, const Satellite& s) { return s.cpuCapacity;                            } },
        { "sat.networkCapacity",   [](const Job&, const Satellite& s) { return s.networkCapacity;                        } },
        { "sat.energyReserved",    [](const Job&, const Satellite& s) { return s.energyReserved;                         } },
        { "sat.remainingEnergy",   [](const Job&, const Satellite& s) { return s.remainingEnergy;                        } },
        { "sat.loadRatio",         [](const Job&, const Satellite& s) {
            return (s.cpuCapacity > 0.0)
                ? s.tasks.size() / s.cpuCapacity
                : 0.0;
        }},
    };
    return reg;
}
#pragma endregion --------------------------------------------------------------