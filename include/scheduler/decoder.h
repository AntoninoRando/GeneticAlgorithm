#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <vector>
#include "types.h"
#include "registry.h"
#include "expr_tree.h"

using namespace std;
// -----------------------------------------------------------------------------



// ─────────────────────────────────────────────────────────────────────────────
//  GREEDY DECODER
//  Given a priority tree, simulate a greedy schedule for the fixed set of jobs.
// ─────────────────────────────────────────────────────────────────────────────

struct ScheduleResult {
    int    scheduledJobs  = 0;
    double servedPriority = 0.0;
    int    conflicts      = 0;
    int    lateness       = 0;
    double fitness        = -numeric_limits<double>::infinity();
};

class GreedyDecoder {
public:
    static constexpr int MIN_GAP = 3;



    #pragma region CONSTRUCTORS ------------------------------------------------
    GreedyDecoder(const vector<Satellite>& satellites, const vector<Job>& jobs,
                  const vector<TerminalDef>& registry)
        : satellites_(satellites), jobs_(jobs), registry_(registry) {}
    #pragma endregion ----------------------------------------------------------

    

    ScheduleResult decode(const ExprNode& tree) const {
        // 1. Score each job using the tree (use the satellite of its best opportunity).
        vector<pair<double, int>> scored; // (score, jobIndex)
        scored.reserve(jobs_.size());
        for (int j = 0; j < static_cast<int>(jobs_.size()); ++j) {
            if (jobs_[j].opportunities.empty()) { scored.emplace_back(-1e18, j); continue; }
            // Score against the satellite of the first available opportunity
            // (a simple proxy; could also take max over all opportunities).
            const Satellite& sat = satellites_[jobs_[j].opportunities[0].satelliteId];
            double score = tree.eval(jobs_[j], sat, registry_);
            if (!isfinite(score)) score = -1e18;
            scored.emplace_back(score, j);
        }

        // Sort jobs by descending score (highest priority first).
        sort(scored.begin(), scored.end(), [](const auto& a, const auto& b){
            return a.first > b.first;
        });

        // 2. Greedy assignment: for each job (in priority order), try its
        //    opportunities and pick the earliest one that fits.
        // satUsed[s] = sorted list of (start, end) already assigned on satellite s.
        vector<vector<pair<int,int>>> satUsed(satellites_.size());

        ScheduleResult res;
        vector<int> assignment(jobs_.size(), -1); // assigned opportunity index per job

        for (const auto& [score, j] : scored) {
            const Job& job = jobs_[j];
            int bestOpp = -1;
            int bestStart = INT_MAX;

            for (int o = 0; o < static_cast<int>(job.opportunities.size()); ++o) {
                const Opportunity& op = job.opportunities[o];
                if (!fits(satUsed[op.satelliteId], op.startMinute, op.endMinute)) continue;
                if (op.startMinute < bestStart) {
                    bestStart = op.startMinute;
                    bestOpp   = o;
                }
            }

            if (bestOpp < 0) continue; // no feasible window

            const Opportunity& chosen = job.opportunities[bestOpp];
            assignment[j] = bestOpp;
            insertSorted(satUsed[chosen.satelliteId], chosen.startMinute, chosen.endMinute);

            res.scheduledJobs++;
            res.servedPriority += job.priority;
            res.lateness += max(0, chosen.endMinute - job.dueMinute);
        }

        // 3. Count gap violations post-hoc (for diagnostics; decoder already avoids overlaps).
        for (auto& intervals : satUsed) {
            for (size_t i = 1; i < intervals.size(); ++i) {
                int gap = intervals[i].first - intervals[i-1].second;
                if      (gap < 0)       res.conflicts += 2;
                else if (gap < MIN_GAP) res.conflicts += 1;
            }
        }

        const int unscheduled = static_cast<int>(jobs_.size()) - res.scheduledJobs;
        res.fitness = 40.0 * res.servedPriority
                    +  6.0 * res.scheduledJobs
                    - 80.0 * res.conflicts
                    -  0.6 * res.lateness
                    - 12.0 * unscheduled;

        return res;
    }

private:
    const vector<Satellite>&   satellites_;
    const vector<Job>&         jobs_;
    const vector<TerminalDef>& registry_;

    // Can the interval [s,e) be inserted without overlap or gap < MIN_GAP?
    static bool fits(const vector<pair<int,int>>& used, int s, int e) {
        for (const auto& [us, ue] : used) {
            if (s < ue && e > us) return false;          // overlap
            if (abs(s - ue) < MIN_GAP) return false;     // gap after existing
            if (abs(us - e) < MIN_GAP) return false;     // gap before existing
        }
        return true;
    }

    static void insertSorted(vector<pair<int,int>>& used, int s, int e) {
        used.emplace_back(s, e);
        sort(used.begin(), used.end());
    }
};

