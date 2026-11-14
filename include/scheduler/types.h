#pragma once

#include <ctime>
#include <vector>
#include <limits>
#include <string>
#include <unordered_set>

using namespace std;
// -----------------------------------------------------------------------------



struct Job {
    int id;
    int arrivalTime;
    string type;
    double ram;
    double disk;
    double imageSize;
    int executionTime;
    int transferTime;
    int numberOfHops;
    string executionServerName;
};

struct Satellite {
    int id;
    string name;

    /// @brief  Simulation minute at which this satellite enters its orbital
    ///         sunset and becomes unavailable.  Any job whose completion time
    ///         would exceed this value is rejected for this satellite.
    ///         Set to a large value (e.g. numeric_limits<double>::max()) to
    ///         model a satellite that is always available within the horizon.
    double operationalUntil = numeric_limits<double>::max();

    bool isAccessPoint;
    double elevationAngle;
    vector<int> neighbors;
    double latency;
    double bandwidth;
    int completedTasks;

    /// @brief  Simulation minute until which the satellite's CPU is already
    ///         committed.  In the data this is always 0 (no backlog at the
    ///         snapshot).  When FitnessEvaluator::trackBusyUntil is on, the
    ///         greedy decoder writes the completion minute of the last job it
    ///         commits here, which makes the `sat.cpuBusyUntil` terminal a
    ///         live "when is this satellite free?" feature (the same quantity
    ///         as the instance available time IAT in Sun et al. 2024, Table 2,
    ///         and as the buffer load CL in Zhang & Yang 2025, Table 2).
    ///         double, not int: completion minutes are not guaranteed to be
    ///         whole numbers, and truncating them would blur the feature.
    double cpuBusyUntil = 0.0;
    int networkBusyUntil;
    double cpuCapacity;
    double networkCapacity;
    double energyReserved;
    vector<int> rejectedTasks;
    double remainingEnergy;
    vector<int> tasks;
    vector<int> deadTasks;
};