#pragma once

#include <vector>
#include <limits>
#include <string>
#include <functional>
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
*/



#pragma region TYPES -----------------------------------------------------------
using FieldAccessor = function<double(const Job&, const Satellite&)>;

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