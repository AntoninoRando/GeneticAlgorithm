#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "expr_tree.h"
#include "registry.h"
#include "scheduler_gp.h"
#include "types.h"

using namespace std;
// -----------------------------------------------------------------------------



// =============================================================================
//  fitness_evaluator.h
//
//  Provides two public-facing types:
//
//    SimulationSnapshot   –  a point-in-time view of the world that the
//                            external app pushes into the evaluator before
//                            each GP generation.
//
//    FitnessEvaluator     –  runs every GPIndividual's expression tree as a
//                            greedy scheduling priority function against the
//                            snapshot, then writes a scalar fitness value back
//                            into each individual.
//
//  FITNESS OBJECTIVE
//  -----------------
//  We want to *minimise* two costs:
//    1. Mean response time   – average (completionMinute − arrivalTime) across
//                              all jobs that were successfully scheduled.
//    2. Energy cost          – total energy consumed by all satellite-job
//                              assignments.
//
//  Both costs are combined into a single maximisation fitness so that the GP
//  framework (which maximises) works without modification:
//
//    fitness = w_jobs  * completionRatio          (reward)
//            − w_time  * normalisedMeanResponse   (penalty)
//            − w_energy* normalisedEnergySpent    (penalty)
//
//  where every term is normalised to [0, 1] across the current population so
//  that no single dimension dominates due to scale differences.
// =============================================================================



// ─────────────────────────────────────────────────────────────────────────────
//  SimulationSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// @brief  A self-contained, point-in-time description of the world state that
///         the external application must fill and pass to FitnessEvaluator
///         before asking it to score a GP generation.
///
/// All satellite and job objects are **copied** into the snapshot so that the
/// evaluator can modify working copies freely without touching the caller's
/// data.
struct SimulationSnapshot {

    /// @brief  The simulation clock at which this snapshot was taken, expressed
    ///         in minutes from the start of the simulation epoch.  Used to
    ///         derive urgency-related terminal values such as remaining
    ///         deadline.
    double currentMinute = 0.0;

    /// @brief  Current state of every satellite in the constellation.
    ///         Each satellite will be used as a candidate resource when
    ///         the decoder tries to assign jobs.
    vector<Satellite> satellites;

    /// @brief  Pool of jobs that are available for scheduling at this instant.
    ///         Jobs whose arrivalTime > currentMinute are ignored by the
    ///         decoder (they have not arrived yet).
    vector<Job> jobs;

    /// @brief  Energy consumed per minute of active computation by satellite
    ///         index (must be the same size as `satellites`).  If left empty
    ///         the evaluator assumes a uniform cost of 1.0 per satellite.
    vector<double> energyPerMinute;

    // ── Convenience constructor ───────────────────────────────────────────
    /// @brief  Construct a snapshot, optionally providing a uniform energy
    ///         rate for every satellite.
    /// @param  currentMinute_   Simulation clock (minutes).
    /// @param  satellites_      Current satellite states.
    /// @param  jobs_            Available job pool.
    /// @param  uniformEnergy    Energy consumed per minute per satellite.
    ///                          Set to 0 to use the per-satellite vector.
    SimulationSnapshot(double          currentMinute_,
                       vector<Satellite> satellites_,
                       vector<Job>       jobs_,
                       double            uniformEnergy = 1.0)
        : currentMinute(currentMinute_)
        , satellites(move(satellites_))
        , jobs(move(jobs_))
    {
        if (uniformEnergy > 0.0) {
            energyPerMinute.assign(this->satellites.size(), uniformEnergy);
        }
    }

    SimulationSnapshot() = default;
};



// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers  (not part of the public API)
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

/// @brief  Working state for one satellite during a single decoding run.
///         Tracks when the satellite becomes free again and how much energy
///         it has consumed.
struct SatState {
    int    satIndex     = 0;    ///< Index back into the working satellite list.
    double freeAtMinute = 0.0;  ///< Earliest minute this satellite can start
                                ///<  a new job.
    double energySpent  = 0.0;  ///< Total energy consumed so far.
};

/// @brief  Record of a single successful job assignment produced by the
///         decoder.
struct Assignment {
    int    jobIndex;            ///< Index into the working job list.
    int    satIndex;            ///< Index into the working satellite list.
    double startMinute;         ///< Minute at which execution begins.
    double completionMinute;    ///< Minute at which execution finishes.
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief  Greedy decoder: uses `tree` as a priority function to assign jobs
///         to satellites, producing a complete schedule.
///
/// Algorithm
/// ---------
/// 1. Build a mutable copy of satellites and jobs from the snapshot.
/// 2. While unscheduled jobs remain:
///    a. For every (unscheduled job, satellite) pair evaluate the tree.
///    b. Pick the pair with the *highest* score.
///    c. Assign that job to that satellite: update the satellite's
///       `freeAtMinute` and `activeTasks`; mark the job as scheduled.
/// 3. Return the list of assignments plus per-satellite energy totals.
///
/// @param  tree        Expression tree used as a priority function.
/// @param  registry    Terminal registry shared with the tree builder.
/// @param  snap        Read-only snapshot of the simulation state.
/// @param  assignments Output: one record per successfully scheduled job.
/// @param  satStates   Output: energy and timeline per satellite.
// ─────────────────────────────────────────────────────────────────────────────
inline void greedyDecode(const ExprNode&               tree,
                          const vector<TerminalDef>&    registry,
                          const SimulationSnapshot&     snap,
                          vector<Assignment>&           assignments,
                          vector<SatState>&             satStates)
{
    // ── Working copies ────────────────────────────────────────────────────
    // We mutate activeTasks / freeAtMinute locally so that each individual
    // gets an identical, clean world.

    vector<Satellite> sats = snap.satellites;
    vector<Job>       jobs = snap.jobs;

    const int nSat = static_cast<int>(sats.size());
    const int nJob = static_cast<int>(jobs.size());

    satStates.resize(nSat);
    for (int s = 0; s < nSat; ++s) {
        satStates[s].satIndex     = s;
        satStates[s].freeAtMinute = snap.currentMinute;
        satStates[s].energySpent  = 0.0;
    }

    const double epm = snap.energyPerMinute.empty() ? 1.0 : 0.0; // fallback flag

    vector<bool> scheduled(nJob, false);
    int remaining = nJob;

    // Filter out jobs that have not arrived yet.
    for (int j = 0; j < nJob; ++j) {
        if (static_cast<double>(jobs[j].arrivalTime) > snap.currentMinute) {
            scheduled[j] = true;   // treat as unavailable
            --remaining;
        }
    }

    while (remaining > 0) {
        double bestScore = -numeric_limits<double>::infinity();
        int    bestJob   = -1;
        int    bestSat   = -1;

        for (int j = 0; j < nJob; ++j) {
            if (scheduled[j]) continue;

            // Refresh the job's remaining deadline for the tree's benefit.
            jobs[j].remainingDeadline =
                static_cast<double>(jobs[j].dueMinute) - snap.currentMinute;

            for (int s = 0; s < nSat; ++s) {
                // Only consider the satellite if it can finish the job in time.
                double startTime      = satStates[s].freeAtMinute;
                double completionTime = startTime +
                                        static_cast<double>(jobs[j].durationMinutes);

                if (completionTime > static_cast<double>(jobs[j].dueMinute))
                    continue;   // would miss the deadline — skip

                double score = tree.eval(jobs[j], sats[s], registry);

                if (score > bestScore) {
                    bestScore = score;
                    bestJob   = j;
                    bestSat   = s;
                }
            }
        }

        // No feasible (job, satellite) pair found — stop early.
        if (bestJob == -1) break;

        // ── Commit assignment ─────────────────────────────────────────────
        double start      = satStates[bestSat].freeAtMinute;
        double duration   = static_cast<double>(jobs[bestJob].durationMinutes);
        double completion = start + duration;

        Assignment a;
        a.jobIndex        = bestJob;
        a.satIndex        = bestSat;
        a.startMinute     = start;
        a.completionMinute= completion;
        assignments.push_back(a);

        // Update satellite timeline and energy.
        satStates[bestSat].freeAtMinute = completion;
        sats[bestSat].activeTasks      += 1;

        double rate = snap.energyPerMinute.empty() ? 1.0 : snap.energyPerMinute[bestSat];
        satStates[bestSat].energySpent += rate * duration;

        scheduled[bestJob] = true;
        --remaining;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
/// @brief  Given a completed decoder run, compute the three raw objective
///         values used for fitness normalisation.
///
/// @param  assignments   Output of greedyDecode.
/// @param  satStates     Output of greedyDecode.
/// @param  jobs          Job list (to read arrivalTime).
/// @param  totalJobs     Total number of jobs in the snapshot (including
///                       not-yet-arrived ones, so the ratio is meaningful).
/// @param  outCompRatio  Fraction of jobs successfully scheduled ∈ [0,1].
/// @param  outMeanResp   Mean response time (minutes) of scheduled jobs.
/// @param  outEnergy     Total energy consumed across all satellites.
// ─────────────────────────────────────────────────────────────────────────────
inline void computeRawObjectives(const vector<Assignment>& assignments,
                                  const vector<SatState>&   satStates,
                                  const vector<Job>&        jobs,
                                  int                       totalJobs,
                                  double&                   outCompRatio,
                                  double&                   outMeanResp,
                                  double&                   outEnergy)
{
    const int scheduled = static_cast<int>(assignments.size());

    outCompRatio = (totalJobs > 0)
                    ? static_cast<double>(scheduled) / static_cast<double>(totalJobs)
                    : 0.0;

    double totalResponse = 0.0;
    for (const auto& a : assignments) {
        double arrival  = static_cast<double>(jobs[a.jobIndex].arrivalTime);
        totalResponse  += a.completionMinute - arrival;
    }
    outMeanResp = (scheduled > 0) ? totalResponse / static_cast<double>(scheduled)
                                  : 0.0;

    outEnergy = 0.0;
    for (const auto& ss : satStates) outEnergy += ss.energySpent;
}

} // namespace detail



// ─────────────────────────────────────────────────────────────────────────────
//  FitnessEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// @brief  Evaluates every individual in a GP population by simulating a
///         greedy schedule with each individual's expression tree as the
///         priority function, then writes a scalar fitness score back into
///         each GPIndividual::fitness field.
///
/// Usage
/// -----
/// @code
///     FitnessEvaluator evaluator(buildTerminalRegistry());
///
///     // Before each generation:
///     SimulationSnapshot snap(currentMinute, satellites, jobs);
///     evaluator.evaluate(gp.getPopulation(), snap);
///
///     // GP can now evolve:
///     gp.solveNextGeneration();
/// @endcode
///
/// Fitness formula
/// ---------------
/// Raw objectives collected for every individual:
///   - `compRatio`  ∈ [0,1]  – fraction of feasible jobs scheduled (maximise)
///   - `meanResp`   ≥ 0      – average response time in minutes   (minimise)
///   - `energy`     ≥ 0      – total energy consumed               (minimise)
///
/// After all individuals are decoded, each objective is normalised to [0,1]
/// across the population.  The final fitness is:
///
///   fitness = w_jobs   * compRatio_norm
///           − w_time   * meanResp_norm
///           − w_energy * energy_norm
///
/// with default weights  w_jobs=0.6,  w_time=0.25,  w_energy=0.15.
class FitnessEvaluator {
public:

    // ── Weights ───────────────────────────────────────────────────────────

    /// @brief  Weight applied to the completion-ratio term (reward).
    ///         Higher values push the GP to schedule more jobs.
    double weightJobs   = 0.60;

    /// @brief  Weight applied to the mean-response-time penalty.
    ///         Higher values push the GP to complete jobs faster.
    double weightTime   = 0.25;

    /// @brief  Weight applied to the energy-consumption penalty.
    ///         Higher values push the GP to prefer energy-efficient assignments.
    double weightEnergy = 0.15;



    // ── Constructors ──────────────────────────────────────────────────────

    /// @brief  Construct the evaluator with the terminal registry produced by
    ///         buildTerminalRegistry().
    /// @param  registry   Terminal registry shared with the GP's TreeBuilder.
    explicit FitnessEvaluator(vector<TerminalDef> registry)
        : registry_(move(registry)) {}



    // ── Public API ────────────────────────────────────────────────────────

    /// @brief  Score every individual in `population` using `snap` as the
    ///         simulation environment, and write the result into each
    ///         individual's `fitness` field.
    ///
    /// This is the only method the GP main loop needs to call.
    ///
    /// @param  population  The current GP population (modified in place).
    /// @param  snap        Read-only snapshot of the current simulation state.
    void evaluate(vector<GPIndividual>& population,
                  const SimulationSnapshot& snap)
    {
        const int n = static_cast<int>(population.size());
        if (n == 0 || snap.jobs.empty() || snap.satellites.empty()) return;

        // ── Step 1: decode every individual ──────────────────────────────
        vector<double> compRatios(n), meanResps(n), energies(n);

        for (int i = 0; i < n; ++i) {
            vector<detail::Assignment> assignments;
            vector<detail::SatState>   satStates;

            detail::greedyDecode(population[i].tree,
                                  registry_,
                                  snap,
                                  assignments,
                                  satStates);

            detail::computeRawObjectives(assignments,
                                          satStates,
                                          snap.jobs,
                                          static_cast<int>(snap.jobs.size()),
                                          compRatios[i],
                                          meanResps[i],
                                          energies[i]);
        }

        // ── Step 2: normalise penalty objectives only ─────────────────────────
        auto normalise = [&](vector<double>& v) {
            double lo    = *min_element(v.begin(), v.end());
            double hi    = *max_element(v.begin(), v.end());
            double range = hi - lo;
            if (range < 1e-12) {
                // Uniform across population: zero out the penalty (no info to act on)
                fill(v.begin(), v.end(), 0.0);
            } else {
                for (auto& x : v) x = (x - lo) / range;
            }
        };

        // compRatio stays absolute — preserves selection pressure when the
        // whole population ties (everyone gets their true [0,1] reward).
        // Only the cost terms are population-relative.
        normalise(meanResps);
        normalise(energies);

        // ── Step 3: write fitness ─────────────────────────────────────────────
        for (int i = 0; i < n; ++i) {
            population[i].fitness =
                weightJobs   * compRatios[i]   // absolute ∈ [0,1]
                - weightTime   * meanResps[i]    // normalised ∈ [0,1]
                - weightEnergy * energies[i];    // normalised ∈ [0,1]
        }
    }

    /// @brief  Evaluate a *single* individual and return its raw (un-normalised)
    ///         objectives.  Useful for debugging or logging the best individual
    ///         after evolution ends.
    ///
    /// @param  individual      The GP individual to evaluate.
    /// @param  snap            Simulation snapshot.
    /// @param  outCompRatio    Output: fraction of jobs scheduled.
    /// @param  outMeanResp     Output: mean response time (minutes).
    /// @param  outEnergy       Output: total energy consumed.
    void evaluateSingle(const GPIndividual&       individual,
                        const SimulationSnapshot& snap,
                        double&                   outCompRatio,
                        double&                   outMeanResp,
                        double&                   outEnergy) const
    {
        vector<detail::Assignment> assignments;
        vector<detail::SatState>   satStates;

        detail::greedyDecode(individual.tree,
                              registry_,
                              snap,
                              assignments,
                              satStates);

        detail::computeRawObjectives(assignments,
                                      satStates,
                                      snap.jobs,
                                      static_cast<int>(snap.jobs.size()),
                                      outCompRatio,
                                      outMeanResp,
                                      outEnergy);
    }

    /// @brief  Return the terminal registry held by this evaluator.
    const vector<TerminalDef>& registry() const { return registry_; }



private:
    vector<TerminalDef> registry_;
};