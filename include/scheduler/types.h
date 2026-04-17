#pragma once

#include <ctime>
#include <vector>
#include <limits>
#include <string>
#include <unordered_set>

using namespace std;
// -----------------------------------------------------------------------------



struct Job {
    int    id;
    string name;
    
    int    durationMinutes;
    int    dueMinute;
    double priority;
    time_t arrivalTime      = 0;
    double taskSize         = 0.0;
    double initialDeadline  = 0.0;
    double remainingDeadline = 0.0;
};

struct Satellite {
    int    id;
    string name;

    /// @brief Set of satellites that this satellite can communicate with.
    vector<Satellite> listeningDome;

    int    activeTasks          = 0;
    double RemainingEnergy      = 0.0;
    double ComputingLoad        = 0.0;
    double ComputingCapability  = 0.0;
};