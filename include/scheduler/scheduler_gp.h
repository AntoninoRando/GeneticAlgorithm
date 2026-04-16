

#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "decoder.h"
#include "registry.h"
#include "tree_builder.h"
#include "tree_operators.h"

using namespace std;

#pragma region TYPES -----------------------------------------------------------
struct GPIndividual {
    ExprNode       tree;
    ScheduleResult result;

    bool operator<(const GPIndividual& o) const {
        return result.fitness < o.result.fitness;
    }
};
#pragma endregion --------------------------------------------------------------



#pragma region ALGORITHM -------------------------------------------------------

class SchedulerGP {
public:
    SchedulerGP(vector<Satellite> satellites, vector<Job> jobs)
        : satellites_(move(satellites))
        , jobs_(move(jobs))
        , rng_(random_device{}())
        , registry_(buildTerminalRegistry())
        , builder_(registry_, rng_)
        , decoder_(satellites_, jobs_, registry_)
        , ops_(rng_, builder_, /*maxDepth=*/7)
    {
        ops_.setTerminalCount(static_cast<int>(registry_.size()));
    }

    // ── Main entry point ─────────────────────────────────────────────────────
    GPIndividual solve(int populationSize  = 200,
                       int generations     = 300,
                       double crossoverRate = 0.80,
                       double mutationRate  = 0.08)
    {
        // Initialise population with ramped half-and-half.
        vector<ExprNode> seeds = builder_.rampedHalfAndHalf(populationSize, 2, 6);
        vector<GPIndividual> population(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            population[i].tree   = move(seeds[i]);
            population[i].result = decoder_.decode(population[i].tree);
        }

        GPIndividual best = *max_element(population.begin(), population.end());

        for (int gen = 0; gen < generations; ++gen) {

            sort(population.begin(), population.end()); // ascending fitness

            vector<GPIndividual> next;
            next.reserve(populationSize);

            // Elitism: keep top 2.
            next.push_back(population.back());
            if (populationSize > 1) next.push_back(population[population.size() - 2]);

            while (static_cast<int>(next.size()) < populationSize) {
                const GPIndividual& pA = tournamentSelect(population, 5);
                const GPIndividual& pB = tournamentSelect(population, 5);

                GPIndividual childA, childB;

                double roll = randomReal();
                if (roll < crossoverRate) {
                    auto [tA, tB] = ops_.crossover(pA.tree, pB.tree);
                    childA.tree   = move(tA);
                    childB.tree   = move(tB);
                } else {
                    childA.tree = pA.tree;
                    childB.tree = pB.tree;
                }

                // Mutation
                ops_.mutate(childA.tree, mutationRate);
                // Occasional hoist to fight bloat.
                if (randomReal() < 0.05) ops_.hoist(childA.tree);

                ops_.mutate(childB.tree, mutationRate);
                if (randomReal() < 0.05) ops_.hoist(childB.tree);

                childA.result = decoder_.decode(childA.tree);
                childB.result = decoder_.decode(childB.tree);

                next.push_back(move(childA));
                if (static_cast<int>(next.size()) < populationSize)
                    next.push_back(move(childB));
            }

            population = move(next);

            const GPIndividual& genBest = *max_element(population.begin(), population.end());
            if (genBest.result.fitness > best.result.fitness) best = genBest;

            if ((gen + 1) % 50 == 0 || gen == generations - 1) {
                cout << "Gen " << setw(4) << gen + 1
                     << " | Fitness: "  << fixed << setprecision(2) << best.result.fitness
                     << " | Jobs: "     << best.result.scheduledJobs << "/" << jobs_.size()
                     << " | Priority: " << best.result.servedPriority
                     << " | Conflicts: "<< best.result.conflicts
                     << " | Late: "     << best.result.lateness
                     << " | TreeSize: " << best.tree.size() << "\n";
            }
        }

        return best;
    }

    // ── Decode and print the schedule produced by the best tree ──────────────
    void printSchedule(const GPIndividual& best) const {
        cout << "\n=== Best Schedule (GP Tree Scheduler) ===\n";
        cout << "Fitness:       " << fixed << setprecision(2) << best.result.fitness  << "\n";
        cout << "Scheduled:     " << best.result.scheduledJobs << " / " << jobs_.size() << "\n";
        cout << "Served prio:   " << best.result.servedPriority << "\n";
        cout << "Conflicts:     " << best.result.conflicts << "\n";
        cout << "Total lateness:" << best.result.lateness << " min\n";
        cout << "Tree size:     " << best.tree.size() << " nodes\n\n";

        cout << "Priority expression:\n  " << best.tree.toString(registry_) << "\n\n";

        // Re-decode to get per-job assignment details.
        vector<pair<double, int>> scored;
        scored.reserve(jobs_.size());
        for (int j = 0; j < static_cast<int>(jobs_.size()); ++j) {
            if (jobs_[j].opportunities.empty()) { scored.emplace_back(-1e18, j); continue; }
            const Satellite& sat = satellites_[jobs_[j].opportunities[0].satelliteId];
            double sc = best.tree.eval(jobs_[j], sat, registry_);
            scored.emplace_back(isfinite(sc) ? sc : -1e18, j);
        }
        sort(scored.begin(), scored.end(), [](const auto& a, const auto& b){ return a.first > b.first; });

        vector<vector<pair<int,int>>> satUsed(satellites_.size());
        vector<int> assignment(jobs_.size(), -1);
        for (const auto& [sc, j] : scored) {
            for (int o = 0; o < static_cast<int>(jobs_[j].opportunities.size()); ++o) {
                const Opportunity& op = jobs_[j].opportunities[o];
                bool ok = true;
                for (const auto& [us, ue] : satUsed[op.satelliteId]) {
                    if (op.startMinute < ue && op.endMinute > us) { ok = false; break; }
                    if (abs(op.startMinute - ue) < 3) { ok = false; break; }
                    if (abs(us - op.endMinute)   < 3) { ok = false; break; }
                }
                if (ok) {
                    assignment[j] = o;
                    satUsed[op.satelliteId].emplace_back(op.startMinute, op.endMinute);
                    sort(satUsed[op.satelliteId].begin(), satUsed[op.satelliteId].end());
                    break;
                }
            }
        }

        cout << left << setw(12) << "Job"
             << setw(12) << "Satellite"
             << setw(8)  << "Score"
             << setw(12) << "Start"
             << setw(12) << "End"
             << setw(10) << "Due"
             << setw(8)  << "Late" << "\n";
        cout << string(74, '-') << "\n";

        for (int j = 0; j < static_cast<int>(jobs_.size()); ++j) {
            double sc = 0.0;
            for (const auto& [s, idx] : scored) if (idx == j) { sc = s; break; }

            if (assignment[j] < 0) {
                cout << left << setw(12) << jobs_[j].name
                     << setw(12) << "UNSCHEDULED"
                     << setw(8)  << fixed << setprecision(1) << sc
                     << "\n";
                continue;
            }
            const Opportunity& op = jobs_[j].opportunities[assignment[j]];
            int late = max(0, op.endMinute - jobs_[j].dueMinute);
            cout << left << setw(12) << jobs_[j].name
                 << setw(12) << satellites_[op.satelliteId].name
                 << setw(8)  << fixed << setprecision(1) << sc
                 << setw(12) << op.startMinute
                 << setw(12) << op.endMinute
                 << setw(10) << jobs_[j].dueMinute
                 << setw(8)  << late << "\n";
        }
    }

    // Expose registry for external inspection.
    const vector<TerminalDef>& registry() const { return registry_; }

private:
    vector<Satellite>    satellites_;
    vector<Job>          jobs_;
    mt19937              rng_;
    vector<TerminalDef>  registry_;
    TreeBuilder          builder_;
    GreedyDecoder        decoder_;
    TreeOperators        ops_;

    double randomReal() {
        uniform_real_distribution<double> d{0.0, 1.0};
        return d(rng_);
    }
    int randomInt(int lo, int hi) {
        uniform_int_distribution<int> d{lo, hi};
        return d(rng_);
    }

    const GPIndividual& tournamentSelect(const vector<GPIndividual>& pop, int k) {
        int best = randomInt(0, static_cast<int>(pop.size()) - 1);
        for (int i = 1; i < k; ++i) {
            int cand = randomInt(0, static_cast<int>(pop.size()) - 1);
            if (pop[cand].result.fitness > pop[best].result.fitness) best = cand;
        }
        return pop[best];
    }
};

#pragma endregion