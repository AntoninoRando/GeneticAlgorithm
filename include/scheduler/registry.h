#pragma once

#include <vector>
#include <limits>
#include <string>
#include <functional>
#include "types.h"

using namespace std;
// -----------------------------------------------------------------------------



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
    // ── Job fields ────────────────────────────────────────────────────────
    vector<TerminalDef> reg = {
        { "job.durationMinutes",   [](const Job& j, const Satellite&) { return static_cast<double>(j.durationMinutes);   } },
        { "job.dueMinute",         [](const Job& j, const Satellite&) { return static_cast<double>(j.dueMinute);         } },
        { "job.priority",          [](const Job& j, const Satellite&) { return j.priority;                               } },
        { "job.arrivalTime",       [](const Job& j, const Satellite&) { return static_cast<double>(j.arrivalTime);       } },
        { "job.taskSize",          [](const Job& j, const Satellite&) { return j.taskSize;                               } },
        { "job.initialDeadline",   [](const Job& j, const Satellite&) { return j.initialDeadline;                        } },
        { "job.remainingDeadline", [](const Job& j, const Satellite&) { return j.remainingDeadline;                      } },
        { "job.opportunityCount",  [](const Job& j, const Satellite&) { return static_cast<double>(j.opportunities.size()); } },
    // ── Satellite fields ──────────────────────────────────────────────────
        { "sat.activeTasks",         [](const Job&, const Satellite& s) { return static_cast<double>(s.activeTasks);       } },
        { "sat.RemainingEnergy",     [](const Job&, const Satellite& s) { return s.RemainingEnergy;                        } },
        { "sat.ComputingLoad",       [](const Job&, const Satellite& s) { return s.ComputingLoad;                          } },
        { "sat.ComputingCapability", [](const Job&, const Satellite& s) { return s.ComputingCapability;                    } },
    // ── Derived / cross-domain signals ────────────────────────────────────
        { "job.slack",             [](const Job& j, const Satellite&) {
            // slack = dueMinute – durationMinutes  (static urgency proxy)
            return static_cast<double>(j.dueMinute - j.durationMinutes);
        }},
        { "sat.loadRatio",         [](const Job&, const Satellite& s) {
            return (s.ComputingCapability > 0.0)
                ? s.ComputingLoad / s.ComputingCapability
                : 0.0;
        }},
    };
    // ── Adding a new field is ONE line here, e.g.:
    // reg.push_back({ "job.myNewField", [](const Job& j, const Satellite&){ return j.myNewField; } });
    return reg;
}
#pragma endregion --------------------------------------------------------------