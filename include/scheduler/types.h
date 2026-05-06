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

    string orbitalSunset;
    bool isAccessPoint;
    double elevationAngle;
    vector<int> neighbors;
    double latency;
    double bandwidth;
    int completedTasks;
    int cpuBusyUntil;
    int networkBusyUntil;
    double cpuCapacity;
    double networkCapacity;
    double energyReserved;
    vector<int> rejectedTasks;
    double remainingEnergy;
    vector<int> tasks;
    vector<int> deadTasks;
};