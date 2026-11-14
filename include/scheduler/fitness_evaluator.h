#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
//
//  LITERATURE NOTES  (keys refer to thesis/references.bib; section numbers are
//  those of the cited papers, see thesis/literature_notes.md)
//  -------------------------------------------------------------------------
//  * Decoder = hyper-heuristic.  The tree is applied by a fixed greedy decoder
//    that produces the schedule; the schedule's quality is the tree's fitness.
//    That is the "meta-algorithm of scheduling heuristics" of Nguyen et al.
//    2017 (nguyen2017gpsurvey §3.2) and the simulator-in-the-loop evaluation of
//    mtgp_multicloud §4.3, zhang2025mtgper §3.1 and gp_fog §V.  Unlike those,
//    the decoder here is *batch*: it sees every arrived job at once and picks
//    the globally best (job, satellite) pair each round, instead of reacting to
//    one decision point at a time (task arrival / processor idle).
//
//  * Hard feasibility filters.  A pair is dropped before its score is read if
//    the satellite would finish after its sunset (`operationalUntil`) or lacks
//    the energy.  This follows Magliarisi et al.: sunset as a *hard* constraint
//    rather than a soft preference (magliarisi2026thesis §6.3.1; the
//    OrbitAware rule of magliarisi2025orbit §5.3 requires the response time to
//    be below the residual visibility), and the energy-sufficiency test of
//    icdcs_sc §III-C.  Filtering is better than letting GP learn the rule:
//    a tree could otherwise schedule a job that can never be delivered.
//    NOT enforced: the per-job deadline of magliarisi2025orbit Eq. 7 (jobs have
//    no deadline field).
//
//  * Objectives.  Completion ratio plays the role of the *task success rate*
//    that zhang2025mtgper (§2.2.3, Eq. 18) and magliarisi2025orbit (§7.1) use as
//    their main metric, but without a deadline test (see above).  Mean response
//    time (completion - arrival) is the objective of magliarisi2025orbit
//    (Eq. 6).  Energy follows the budget model of icdcs_sc §II-E.
//
//  * Weighted sum with population-relative normalisation.  The papers
//    optimise ONE objective (success rate in zhang2025mtgper, monetary cost in
//    mtgp_multicloud and dtgpam, makespan in gp_fog) and so need no weights.
//    With seconds and joules mixed, Magliarisi's ILP discussion warns that
//    arbitrarily weighting non-homogeneous metrics often leads to biased
//    solutions and offers a lexicographic alternative (primary objective
//    first, second one within a tolerance; magliarisi2026thesis §6.3.2,
//    Hierarchical ILP), while icdcs_sc normalises each term by a FIXED bound
//    (energy by the maximum battery, time by the largest response).  Here the
//    cost terms are min-max normalised across the current population instead,
//    which needs no bounds but makes fitness non-stationary: it is only
//    comparable within one generation of one world (main.py relies on this).
//    A lexicographic fitness is the natural next step if the weights prove hard
//    to defend.
//
//  * Fixed-bound normalisation (opt-in, FitnessEvaluator::fixedBounds).
//    The population-relative fitness has two measured defects (thesis
//    ch. Experiments): response time is averaged over the scheduled jobs only,
//    so scheduling few short jobs is rewarded, and min-max stretches any
//    spread of a cost term to the full [0,1] range, however small (energy
//    differs by about 1% between rules and still moves the fitness by up to
//    w_energy).  With fixedBounds the cost terms are instead divided by bounds
//    that depend on the WORLD ONLY, never on the population:
//
//        R~ = (1/J) * [ sum over scheduled jobs of r_j / Rmax
//                       + number of unscheduled jobs ]
//           = 1 - C * (1 - Rbar / Rmax)                       (one world)
//        E~ = E / Emax
//        fitness = w_jobs * C  -  w_time * R~  -  w_energy * E~
//
//    Sources.  (1) Normalising each objective by fixed lower/upper bounds
//    (utopia value and maximum) so that the weights compare like with like is
//    the standard transformation for weighted sums (Marler & Arora 2004,
//    marler2004survey), and it is what icdcs_sc does in this very setting
//    (energy over the maximum battery, time over the largest response).
//    (2) Charging a job that is NOT scheduled a fixed penalty, added to the
//    time criterion of the scheduled ones, is the objective of "scheduling
//    with rejection" (Shabtay, Gaspar & Kaspi 2013, shabtay2013rejection):
//    here the rejection cost is Rmax, i.e. 1 after normalisation.
//    The particular bounds Rmax and Emax are derived in this project from the
//    decoder's own constraints (detail::computeWorldBounds; proof in
//    thesis/chapters/method.tex), they are not taken from a paper.
//
//    Properties (proved in method.tex): the fitness of a tree no longer
//    depends on the other individuals, so it is comparable across generations
//    and the best fitness cannot decrease under elitism while the worlds stay
//    the same; and r_j <= Rmax for every schedule the decoder can produce, so
//    leaving a job unscheduled is never cheaper than scheduling it.
//    NOT YET MEASURED on the held-out worlds: default stays off until the
//    ablation (thesis/experiments/run_ablation.py, configs "fixed*") is run.
//
//  * Tie-break (opt-in, FitnessEvaluator::earliestFreeTieBreak).  The decoder
//    is an argmax over (job, satellite) pairs, and a tree that does not look at
//    the satellite scores every satellite alike.  By default such ties go to
//    the lowest satellite index, so the rule queues its jobs on satellite 0
//    while the others are idle: an artefact of the scan order, not a decision
//    of the tree.  With the option the tie goes to the satellite that is free
//    earliest, which is how list scheduling places a job (Graham 1966,
//    graham1966bounds) and what the earliest-finish-time and
//    minimum-completion-time rules do (Topcuoglu et al. 2002,
//    topcuoglu2002heft; Braun et al. 2001, braun2001comparison).  For a rule
//    that orders jobs only, and without the sunset and energy filters, every
//    job then completes no later than under the default (Proposition 2,
//    thesis/chapters/method.tex).  With the filters this is not guaranteed.
//    NOT YET MEASURED on the held-out worlds; default stays off.
//
//  * Several worlds per evaluation.  The papers never score an individual on a
//    single instance: dtgpam §V-A evaluates every individual on 5 randomly
//    drawn training instances per generation (yang2025thesis §3.3.1),
//    zhang2025mtgper §4.1 averages 100 simulations per individual, and
//    mtgp_multicloud §5.2 trains on whole scenario sets.  Averaging removes
//    the incentive to overfit one world.  evaluateMany() provides it.
//
//  * Cost.  Fitness evaluation dominates the run time (mtgp_multicloud §4.4:
//    O(N * n * R) per generation for N decisions, n candidates, rule size R),
//    so the score cache and the OpenMP loop below are where speed is won.
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
/// 1. Build mutable working copies of satellites and jobs from the snapshot.
/// 2. Mark jobs that have not arrived yet as unavailable.
/// 3. While unscheduled, available jobs remain:
///    a. For every (unscheduled job, satellite) pair:
///       - Skip if the satellite would finish the job after its
///         `operationalUntil` sunset (orbital window constraint).
///       - Skip if the satellite does not have enough `remainingEnergy` to
///         cover the job's energy cost at the configured rate (energy
///         constraint).
///       - Evaluate the expression tree to obtain a priority score.
///    b. Pick the (job, satellite) pair with the *highest* score.
///    c. Commit the assignment: advance `freeAtMinute`, deduct
///       `remainingEnergy` from the working satellite copy, and accumulate
///       energy in `satStates`.
/// 4. Return the list of assignments and per-satellite energy totals.
///
/// Feasibility constraints
/// -----------------------
///  - **Operational window**: `freeAtMinute + duration ≤ satellite.operationalUntil`
///  - **Energy budget**:      `energyRate * duration  ≤ satellite.remainingEnergy`
///
/// Both constraints are checked before scoring, so unfeasible pairs are never
/// considered regardless of what score the tree would produce.  Jobs for which
/// no feasible satellite exists in a given iteration are left unscheduled and
/// contribute to a lower completion ratio.
///
/// @param  tree        Expression tree used as a priority function.
/// @param  registry    Terminal registry shared with the tree builder.
/// @param  snap        Read-only snapshot of the simulation state.
/// @param  assignments Output: one record per successfully scheduled job.
/// @param  satStates   Output: energy and timeline per satellite.
/// @param  trackBusyUntil  When true, the working satellites' `cpuBusyUntil`
///         is initialised to the snapshot minute and set to each committed
///         job's completion minute, so the `sat.cpuBusyUntil` terminal reports
///         when a satellite becomes free (see types.h).  Off by default
///         because it changes what an existing terminal means.
/// @param  earliestFreeTieBreak  When true, equally scored pairs are ordered
///         by start minute (earliest-free satellite first) instead of by scan
///         order, and scores are compared in a total order (NaN lowest).  See
///         the selection code below and FitnessEvaluator::earliestFreeTieBreak.
// ─────────────────────────────────────────────────────────────────────────────
inline void greedyDecode(const ExprNode&               tree,
                          const vector<TerminalDef>&    registry,
                          const SimulationSnapshot&     snap,
                          vector<Assignment>&           assignments,
                          vector<SatState>&             satStates,
                          bool                          trackBusyUntil = false,
                          bool                          earliestFreeTieBreak = false)
{
    // ── Working state ─────────────────────────────────────────────────────
    // Each individual decoded in the same generation must see an identical,
    // clean world.
    //
    // Jobs are strictly READ-ONLY during a decode, so they are used in place:
    // the previous per-decode copy of snap.jobs allocated two std::strings
    // per job, per individual, per generation, for nothing.
    //
    // Satellites ARE mutated (remainingEnergy / completedTasks — and the
    // expression-tree terminals read those fields, which is why the score
    // cache below refreshes a column per commit), so a working copy is
    // required.  It lives in a thread_local buffer: vector copy-assignment
    // reuses each element's heap storage (name string, neighbor/task
    // vectors), so after the first decode on a thread the copy is
    // allocation-free instead of ~6 heap allocations per satellite.  Safe
    // under OpenMP: each thread owns its own instance, and nothing here
    // escapes the call (outputs go to the caller-owned parameters).
    const vector<Job>& jobs = snap.jobs;
    thread_local vector<Satellite> sats;
    sats = snap.satellites;

    // Live availability terminal (opt-in).  An idle satellite is "free at the
    // snapshot minute", exactly like satStates[s].freeAtMinute below, so the
    // terminal and the decoder agree on the satellite's timeline.
    if (trackBusyUntil)
        for (auto& sat : sats)
            sat.cpuBusyUntil = std::max(sat.cpuBusyUntil, snap.currentMinute);

    const int nSat = static_cast<int>(sats.size());
    const int nJob = static_cast<int>(jobs.size());

    assignments.clear();
    assignments.reserve(nJob);

    satStates.resize(nSat);
    for (int s = 0; s < nSat; ++s) {
        satStates[s].satIndex     = s;
        satStates[s].freeAtMinute = snap.currentMinute;
        satStates[s].energySpent  = 0.0;
        // remainingEnergy is read directly from sats[s] and deducted there
        // as jobs are committed, so no separate mirror is needed here.
    }

    // uint8_t instead of vector<bool>: no bit-packing in the hot scan loop,
    // and the buffer is reused across decodes on the same thread.
    thread_local vector<uint8_t> scheduled;
    scheduled.assign(nJob, 0);
    int remaining = nJob;

    // Jobs whose arrivalTime is in the future are treated as unavailable for
    // this snapshot.  We exclude them up-front rather than inside the inner
    // loop to avoid redundant checks.
    for (int j = 0; j < nJob; ++j) {
        if (static_cast<double>(jobs[j].arrivalTime) > snap.currentMinute) {
            scheduled[j] = 1;
            --remaining;
        }
    }

    // ── Score cache ───────────────────────────────────────────────────────
    // tree.eval(job j, sat s) is a pure function of the job and the satellite.
    // The job objects never change during a decode, and a satellite only
    // changes when a job is *committed* to it (completedTasks += 1 and
    // remainingEnergy -= cost).  Committing therefore invalidates only the
    // scores in that one satellite's column; every other cached score is still
    // exact.  We compute the full J×S matrix once and then refresh a single
    // column per commit, collapsing the original O(J²·S) tree evaluations into
    // O(J·S + J²) — the tree walk is by far the most expensive thing here.
    //
    // The constraint checks (operational window, energy) stay inside the loop:
    // they are cheap and depend on per-iteration satellite state, so the
    // selection below remains byte-for-byte identical to the un-cached version
    // (same scan order, same strict-greater tie-break, same feasibility gate).
    //
    // Layout is row-major in j (index j*nSat + s) so each iteration's argmax
    // scan reads a job's row contiguously.  thread_local so the J×S buffer is
    // allocated once per thread instead of once per decode.
    thread_local vector<double> score;
    score.assign(static_cast<size_t>(nJob) * static_cast<size_t>(nSat), 0.0);
    for (int j = 0; j < nJob; ++j) {
        if (scheduled[j]) continue;                 // never read for these jobs
        const size_t base = static_cast<size_t>(j) * static_cast<size_t>(nSat);
        for (int s = 0; s < nSat; ++s)
            score[base + s] = tree.eval(jobs[j], sats[s], registry);
    }

    while (remaining > 0) {
        double bestScore = -numeric_limits<double>::infinity();
        int    bestJob   = -1;
        int    bestSat   = -1;
        double bestStart = 0.0;     // start minute of the best pair (tie-break)

        for (int j = 0; j < nJob; ++j) {
            if (scheduled[j]) continue;

            const double duration = static_cast<double>(
                jobs[j].executionTime + jobs[j].transferTime);
            const size_t base = static_cast<size_t>(j) * static_cast<size_t>(nSat);

            for (int s = 0; s < nSat; ++s) {
                const double startTime      = satStates[s].freeAtMinute;
                const double completionTime = startTime + duration;

                // ── Constraint 1: operational window ─────────────────────
                // Reject the pair if the satellite would still be executing
                // this job after its orbital sunset.
                if (completionTime > sats[s].operationalUntil) continue;

                // ── Constraint 2: energy budget ───────────────────────────
                // Reject the pair if the satellite does not have enough
                // remaining energy to sustain computation for the full
                // job duration.
                const double rate       = snap.energyPerMinute.empty()
                                            ? 1.0
                                            : snap.energyPerMinute[s];
                const double energyCost = rate * duration;
                if (energyCost > sats[s].remainingEnergy) continue;

                // ── Priority score ────────────────────────────────────────
                // Read the cached tree evaluation instead of recomputing it.
                // The GP maximises this score; both constraints above must
                // pass before the cached score is considered.
                const double sc = score[base + s];
                if (!earliestFreeTieBreak) {
                    // Default selection: a pair wins only with a STRICTLY
                    // higher score, so ties go to the first pair of the scan
                    // (lowest job index, then lowest satellite index).
                    if (sc > bestScore) {
                        bestScore = sc;
                        bestJob   = j;
                        bestSat   = s;
                    }
                } else {
                    // Opt-in selection (FitnessEvaluator::earliestFreeTieBreak):
                    // lexicographic maximum of
                    //     ( score,  -start minute,  -job index,  -satellite index ).
                    // Among equally scored pairs the one that can START first
                    // wins, i.e. the satellite that is free earliest: the
                    // machine choice of list scheduling [graham1966bounds] and
                    // of the earliest-finish-time rules [topcuoglu2002heft,
                    // braun2001comparison] (start and finish order coincide
                    // here, a job lasts the same on every satellite).  The
                    // remaining ties keep the scan order, as before.
                    //
                    // The order is made total: NaN counts as the lowest score
                    // and the first feasible pair is always accepted, so a
                    // pair scoring -inf or NaN is still scheduled when nothing
                    // better is feasible.  (With the strict `>` above such a
                    // pair can never beat the initial -inf, which lets a tree
                    // leave feasible jobs unscheduled.)
                    const double value = (sc != sc)
                        ? -numeric_limits<double>::infinity() : sc;
                    const bool better =
                        bestJob == -1 ||
                        value > bestScore ||
                        (value == bestScore && startTime < bestStart);
                    if (better) {
                        bestScore = value;
                        bestJob   = j;
                        bestSat   = s;
                        bestStart = startTime;
                    }
                }
            }
        }

        // No feasible (job, satellite) pair exists — stop early.
        // Remaining jobs are left unscheduled and lower the completion ratio.
        if (bestJob == -1) break;

        // ── Commit assignment ─────────────────────────────────────────────
        const double start    = satStates[bestSat].freeAtMinute;
        const double duration = static_cast<double>(
            jobs[bestJob].executionTime + jobs[bestJob].transferTime);
        const double completion = start + duration;

        Assignment a;
        a.jobIndex         = bestJob;
        a.satIndex         = bestSat;
        a.startMinute      = start;
        a.completionMinute = completion;
        assignments.push_back(a);

        // Advance the satellite's timeline.
        satStates[bestSat].freeAtMinute = completion;
        sats[bestSat].completedTasks   += 1;
        // Expose it to the tree.  The column refresh further down re-scores the
        // satellite, so every tree that reads `sat.cpuBusyUntil` (or
        // `sat.completedTasks`) sees the new state on the next round.
        if (trackBusyUntil) sats[bestSat].cpuBusyUntil = completion;

        // Deduct energy from the working copy so that subsequent iterations
        // of this decoder run see the reduced budget.  This is the key fix:
        // without it every satellite appears to have unlimited energy.
        const double rate       = snap.energyPerMinute.empty()
                                    ? 1.0
                                    : snap.energyPerMinute[bestSat];
        const double energyCost = rate * duration;
        satStates[bestSat].energySpent    += energyCost;
        sats[bestSat].remainingEnergy     -= energyCost;

        scheduled[bestJob] = 1;
        --remaining;

        // ── Refresh the one invalidated column ─────────────────────────────
        // Only sats[bestSat] changed, so only the (job, bestSat) scores can
        // have changed.  Recompute them for the still-unscheduled jobs (the
        // scores of scheduled jobs are never read again).
        for (int j = 0; j < nJob; ++j) {
            if (scheduled[j]) continue;
            score[static_cast<size_t>(j) * static_cast<size_t>(nSat)
                  + static_cast<size_t>(bestSat)] =
                tree.eval(jobs[j], sats[bestSat], registry);
        }
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
inline void sumObjectives(const vector<Assignment>& assignments,
                          const vector<SatState>&   satStates,
                          const vector<Job>&        jobs,
                          int&                      outScheduled,
                          double&                   outTotalResponse,
                          double&                   outEnergy)
{
    outScheduled = static_cast<int>(assignments.size());

    outTotalResponse = 0.0;
    for (const auto& a : assignments) {
        double arrival    = static_cast<double>(jobs[a.jobIndex].arrivalTime);
        outTotalResponse += a.completionMinute - arrival;
    }

    outEnergy = 0.0;
    for (const auto& ss : satStates) outEnergy += ss.energySpent;
}

inline void computeRawObjectives(const vector<Assignment>& assignments,
                                  const vector<SatState>&   satStates,
                                  const vector<Job>&        jobs,
                                  int                       totalJobs,
                                  double&                   outCompRatio,
                                  double&                   outMeanResp,
                                  double&                   outEnergy)
{
    int    scheduled = 0;
    double totalResponse = 0.0;
    sumObjectives(assignments, satStates, jobs, scheduled, totalResponse, outEnergy);

    outCompRatio = (totalJobs > 0)
                    ? static_cast<double>(scheduled) / static_cast<double>(totalJobs)
                    : 0.0;
    outMeanResp = (scheduled > 0) ? totalResponse / static_cast<double>(scheduled)
                                  : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
/// @brief  Fixed bounds of one world, used by the fixed-bound fitness
///         (FitnessEvaluator::fixedBounds).  They depend on the snapshot only,
///         never on a tree or on the population.
struct WorldBounds {
    /// Upper bound Rmax on the response time (completion - arrival, minutes)
    /// of ANY job in ANY schedule greedyDecode can produce for this world.
    double maxResponse = 0.0;
    /// Upper bound Emax on the total energy of any such schedule.
    double maxEnergy   = 0.0;
};

/// @brief  Computes the bounds of `snap`.  O(J + S).
///
/// Notation: t0 = snapshot minute; a job j is *available* if a_j <= t0 and has
/// duration d_j = executionTime + transferTime; D = sum of d_j over available
/// jobs; a_min = earliest available arrival.  Satellite s has sunset U_s,
/// energy budget B_s and energy rate rho_s.
///
/// The decoder starts every satellite at t0 and runs its jobs back to back
/// (start = freeAtMinute, no idle time), so a job on s completes at
/// t0 + (total duration committed to s so far).  That total is bounded by
///
///     T_s = max(0, min( D,  U_s - t0,  B_s / rho_s ))      (last term only if rho_s > 0)
///
/// (all work there is; the sunset filter; the energy filter, since the
/// energy committed to s never exceeds B_s).  Hence, for every scheduled job,
///
///     r_j = c_j - a_j  <=  (t0 + max_s T_s) - a_min  =:  Rmax,
///     E   = sum_s rho_s * (duration on s)  <=  min( sum_s rho_s T_s,  rho_max * D )  =:  Emax.
///
/// With the shipped worlds the energy filter is the binding one, so
/// Emax = sum of the batteries, the bound icdcs_sc divides by.
///
/// The caller must have validated `snap` (energyPerMinute empty or one entry
/// per satellite).  A world with no available job has both bounds 0.
inline WorldBounds computeWorldBounds(const SimulationSnapshot& snap) {
    WorldBounds b;
    const double t0 = snap.currentMinute;

    double totalDuration   = 0.0;                                   // D
    double earliestArrival = numeric_limits<double>::infinity();    // a_min
    for (const Job& j : snap.jobs) {
        const double arrival = static_cast<double>(j.arrivalTime);
        if (arrival > t0) continue;                                 // not available
        totalDuration  += static_cast<double>(j.executionTime + j.transferTime);
        earliestArrival = std::min(earliestArrival, arrival);
    }
    if (!(earliestArrival <= t0)) return b;                         // nothing to schedule

    double longestBusy = 0.0;       // max_s T_s
    double energyCap   = 0.0;       // sum_s rho_s T_s
    double maxRate     = 0.0;       // rho_max
    for (size_t s = 0; s < snap.satellites.size(); ++s) {
        const Satellite& sat  = snap.satellites[s];
        const double     rate = snap.energyPerMinute.empty() ? 1.0 : snap.energyPerMinute[s];

        double busy = std::min(totalDuration, sat.operationalUntil - t0);
        if (rate > 0.0) busy = std::min(busy, sat.remainingEnergy / rate);
        busy = std::max(busy, 0.0);                                 // T_s

        longestBusy = std::max(longestBusy, busy);
        energyCap  += std::max(rate, 0.0) * busy;
        maxRate     = std::max(maxRate, rate);
    }

    b.maxResponse = (t0 - earliestArrival) + longestBusy;
    b.maxEnergy   = std::min(energyCap, maxRate * totalDuration);
    return b;
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
/// (compRatio is used as is; only the two cost terms are normalised.)
///
/// With `fixedBounds = true` the two cost terms are divided by per-world
/// bounds instead of by population statistics, and unscheduled jobs are
/// charged the worst response; the fitness is then absolute.
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

    /// @brief  Make `sat.cpuBusyUntil` a live terminal: the decoder writes each
    ///         committed job's completion minute into the satellite it used.
    ///         Default false (the terminal is then constantly 0, as before).
    ///         Gives GP the queue-availability signal that is the most used
    ///         routing feature in zhang2025mtgper (§5.2.2: NT, CL) and the
    ///         available-time terminals of mtgp_multicloud (Table 2: AAT, IAT).
    bool trackBusyUntil = false;

    /// @brief  Fixed-bound normalisation of the cost terms (see the "Fixed-bound
    ///         normalisation" note at the top of this file).  Default false:
    ///         the population-relative fitness, exactly as before.
    ///
    ///         When true the fitness is absolute:
    ///           fitness = weightJobs * C - weightTime * R~ - weightEnergy * E~
    ///         with R~ the mean over ALL jobs of r_j / Rmax (an unscheduled job
    ///         counts as 1, the rejection penalty of shabtay2013rejection) and
    ///         E~ = E / Emax, both in [0,1] and independent of the population
    ///         (marler2004survey, icdcs_sc).  Fitness then lies in
    ///         [-(weightTime + weightEnergy), weightJobs].
    bool fixedBounds = false;

    /// @brief  Tie-break of the greedy decoder.  Default false: among equally
    ///         scored feasible pairs the first of the scan wins (lowest job
    ///         index, then lowest satellite index).  A rule that ignores the
    ///         satellite then sends every job to satellite 0 until it is
    ///         infeasible, while the others sit idle ("first-fit").
    ///
    ///         When true the decoder takes the lexicographic maximum of
    ///         (score, -start minute, -job index, -satellite index): among
    ///         equally scored pairs, the satellite that is free earliest.
    ///         That is the machine choice of list scheduling
    ///         (graham1966bounds) and of the earliest-finish-time /
    ///         minimum-completion-time rules (topcuoglu2002heft,
    ///         braun2001comparison).  A strictly higher score still always
    ///         wins: the tree keeps full control, the option only decides
    ///         what the tree leaves undecided.  Scores are also compared in a
    ///         total order (NaN lowest, -inf selectable), so the decoder
    ///         schedules as long as a feasible pair exists.
    ///         Maths and proposition: thesis/chapters/method.tex.
    ///         NOT YET MEASURED on the held-out worlds (run_ablation.py,
    ///         configs "tiebreak*"); the thesis results use the default.
    bool earliestFreeTieBreak = false;



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
        evaluatePointers(population, {&snap});
    }

    /// @brief  Score every individual on SEVERAL worlds at once.
    ///
    /// The objectives are pooled over the worlds before they are normalised:
    ///   - completion ratio = jobs scheduled in all worlds / jobs in all worlds,
    ///   - mean response    = total response of all scheduled jobs / their count,
    ///   - energy           = total energy over the worlds.
    /// With one snapshot this is exactly evaluate().  Degenerate snapshots (no
    /// jobs or no satellites) are skipped; if all are, every fitness is 0.
    ///
    /// Why: see the "Several worlds per evaluation" note at the top of this
    /// file (dtgpam §V-A trains on 5 instances per generation;
    /// zhang2025mtgper §4.1 averages 100 simulations).  Pooling, rather than
    /// averaging per-world ratios, keeps a world with many jobs from being
    /// down-weighted and keeps the "no job scheduled" case well defined.
    void evaluateMany(vector<GPIndividual>& population,
                      const vector<SimulationSnapshot>& snaps)
    {
        vector<const SimulationSnapshot*> ptrs;
        ptrs.reserve(snaps.size());
        for (const auto& sn : snaps) ptrs.push_back(&sn);
        evaluatePointers(population, ptrs);
    }

    /// @brief  Registry indices of terminals whose value is the same for every
    ///         (job, satellite) pair of `snap` AND that the decoder never
    ///         updates.  Those terminals carry no information for the tree and
    ///         can be excluded from the search (SchedulerGP's
    ///         `excludedTerminals`).
    ///
    /// `sat.completedTasks` and `sat.remainingEnergy` are written by the
    /// decoder, and `sat.cpuBusyUntil` is too when trackBusyUntil is on, so
    /// they are kept even if constant in the snapshot.
    vector<int> constantTerminals(const SimulationSnapshot& snap) const {
        vector<int> out;
        if (snap.jobs.empty() || snap.satellites.empty()) return out;
        for (int t = 0; t < static_cast<int>(registry_.size()); ++t) {
            const string& name = registry_[t].name;
            if (name == "sat.completedTasks" || name == "sat.remainingEnergy") continue;
            if (trackBusyUntil && name == "sat.cpuBusyUntil") continue;
            double lo =  numeric_limits<double>::infinity();
            double hi = -numeric_limits<double>::infinity();
            for (const auto& j : snap.jobs)
                for (const auto& sat : snap.satellites) {
                    const double v = registry_[t].accessor(j, sat);
                    lo = min(lo, v);
                    hi = max(hi, v);
                }
            if (hi - lo < 1e-12) out.push_back(t);
        }
        return out;
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
        validateSnapshot(snap);

        vector<detail::Assignment> assignments;
        vector<detail::SatState>   satStates;

        detail::greedyDecode(individual.tree,
                              registry_,
                              snap,
                              assignments,
                              satStates,
                              trackBusyUntil,
                              earliestFreeTieBreak);

        detail::computeRawObjectives(assignments,
                                      satStates,
                                      snap.jobs,
                                      static_cast<int>(snap.jobs.size()),
                                      outCompRatio,
                                      outMeanResp,
                                      outEnergy);
    }

    /// @brief  The fixed bounds (Rmax, Emax) that the fixed-bound fitness
    ///         divides by for `snap`; see detail::computeWorldBounds.  Exposed
    ///         so the normalisation can be inspected and tested.
    static detail::WorldBounds worldBounds(const SimulationSnapshot& snap) {
        validateSnapshot(snap);
        return detail::computeWorldBounds(snap);
    }

    /// @brief  Return the terminal registry held by this evaluator.
    const vector<TerminalDef>& registry() const { return registry_; }



private:
    /// @brief  Shared implementation of evaluate() and evaluateMany().
    ///
    /// Step 1 decodes every individual on every usable snapshot and pools the
    /// raw objectives; steps 2-3 normalise the cost terms across the
    /// population and write the fitness (unchanged from the single-world
    /// version).
    void evaluatePointers(vector<GPIndividual>& population,
                          const vector<const SimulationSnapshot*>& snaps)
    {
        const int n = static_cast<int>(population.size());
        if (n == 0) return;

        // Keep only usable worlds.  A degenerate one (no jobs or no
        // satellites) defines every fitness as 0 when it is the only one,
        // instead of leaving whatever the previous generation (or
        // uninitialised memory) held: callers sort and select on these values.
        vector<const SimulationSnapshot*> use;
        long long totalJobs = 0;
        for (const SimulationSnapshot* sn : snaps) {
            if (sn->jobs.empty() || sn->satellites.empty()) continue;
            validateSnapshot(*sn);
            use.push_back(sn);
            totalJobs += static_cast<long long>(sn->jobs.size());
        }
        if (use.empty()) {
            for (auto& ind : population) ind.fitness = 0.0;
            return;
        }

        // Fixed bounds of every usable world (fixed-bound fitness only).  They
        // are computed once, before the parallel loop, and only read there.
        vector<detail::WorldBounds> bounds;
        double totalMaxEnergy = 0.0;
        if (fixedBounds) {
            bounds.reserve(use.size());
            for (const SimulationSnapshot* sn : use) {
                bounds.push_back(detail::computeWorldBounds(*sn));
                totalMaxEnergy += bounds.back().maxEnergy;
            }
        }

        // ── Step 1: decode every individual ──────────────────────────────
        vector<double> compRatios(n), meanResps(n), energies(n);
        // Fixed-bound response term R~ of each individual (left empty in the
        // population-relative mode, which never reads it).
        vector<double> fixedResps(fixedBounds ? n : 0);

        // Each individual is decoded independently: greedyDecode takes its own
        // working copies of the satellites/jobs, the expression tree and the
        // terminal registry are read-only, and every result is written to a
        // distinct index (compRatios[i] / meanResps[i] / energies[i]). There is
        // therefore no shared mutable state and the loop is data-race free.
        //
        // `assignments` and `satStates` are declared inside the loop body, so
        // OpenMP makes them private to each thread automatically. Runtimes
        // differ per individual (tree size and number of feasible jobs vary),
        // so dynamic scheduling keeps the cores evenly loaded.
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            vector<detail::Assignment> assignments;
            vector<detail::SatState>   satStates;

            long long scheduledSum = 0;
            double    responseSum  = 0.0;
            double    energySum    = 0.0;
            double    normRespSum  = 0.0;   // sum of r_j / Rmax over scheduled jobs

            for (size_t w = 0; w < use.size(); ++w) {
                const SimulationSnapshot* sn = use[w];
                detail::greedyDecode(population[i].tree,
                                      registry_,
                                      *sn,
                                      assignments,
                                      satStates,
                                      trackBusyUntil,
                                      earliestFreeTieBreak);

                int    scheduled = 0;
                double response  = 0.0;
                double energy    = 0.0;
                detail::sumObjectives(assignments, satStates, sn->jobs,
                                      scheduled, response, energy);
                scheduledSum += scheduled;
                responseSum  += response;
                energySum    += energy;
                // Each world is normalised by its OWN bound before pooling, so
                // a world with a long horizon cannot dominate the term.
                if (fixedBounds && bounds[w].maxResponse > 1e-12)
                    normRespSum += response / bounds[w].maxResponse;
            }

            compRatios[i] = static_cast<double>(scheduledSum) / static_cast<double>(totalJobs);
            meanResps[i]  = (scheduledSum > 0)
                                ? responseSum / static_cast<double>(scheduledSum)
                                : 0.0;
            energies[i]   = energySum;

            if (fixedBounds) {
                // R~ = (sum of normalised responses + one unit per unscheduled
                // job) / J.  Jobs that have not arrived count as unscheduled,
                // exactly as they already do in the completion ratio.
                const double unscheduled = static_cast<double>(totalJobs - scheduledSum);
                fixedResps[i] = (normRespSum + unscheduled) / static_cast<double>(totalJobs);
            }
        }

        // ── Fixed-bound fitness (opt-in): absolute, no population statistics ──
        // marler2004survey / icdcs_sc (fixed bounds), shabtay2013rejection
        // (rejection penalty).  The clamps only absorb floating-point drift:
        // both terms are in [0,1] by construction (see computeWorldBounds).
        if (fixedBounds) {
            for (int i = 0; i < n; ++i) {
                const double respTerm   = std::min(1.0, std::max(0.0, fixedResps[i]));
                const double energyTerm = (totalMaxEnergy > 1e-12)
                    ? std::min(1.0, std::max(0.0, energies[i] / totalMaxEnergy))
                    : 0.0;
                population[i].fitness =
                    weightJobs     * compRatios[i]
                    - weightTime   * respTerm
                    - weightEnergy * energyTerm;
            }
            return;
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

        // Mean response: an individual that scheduled ZERO jobs has no
        // response time at all.  It used to report meanResp = 0 — the best
        // possible value — which (a) rewarded empty schedules and (b) dragged
        // the normalisation minimum to 0, inflating everyone else's penalty.
        // Empty schedules are now excluded from the min/max and receive the
        // worst penalty (1.0) instead.
        {
            double lo = numeric_limits<double>::infinity();
            double hi = -numeric_limits<double>::infinity();
            for (int i = 0; i < n; ++i) {
                if (compRatios[i] <= 0.0) continue;
                lo = min(lo, meanResps[i]);
                hi = max(hi, meanResps[i]);
            }
            const double range = hi - lo;   // never read unless some compRatio > 0
            for (int i = 0; i < n; ++i) {
                if (compRatios[i] <= 0.0)   meanResps[i] = 1.0;
                else if (range < 1e-12)     meanResps[i] = 0.0;
                else                        meanResps[i] = (meanResps[i] - lo) / range;
            }
        }

        // Energy: zero consumption is a genuine cost value (scheduling
        // nothing really does cost nothing), so plain min-max stays correct.
        normalise(energies);

        // ── Step 3: write fitness ─────────────────────────────────────────────
        for (int i = 0; i < n; ++i) {
            population[i].fitness =
                weightJobs   * compRatios[i]   // absolute ∈ [0,1]
                - weightTime   * meanResps[i]    // normalised ∈ [0,1]
                - weightEnergy * energies[i];    // normalised ∈ [0,1]
        }
    }

    /// @brief  Rejects malformed snapshots up front.  In particular a
    ///         non-empty energyPerMinute shorter than the satellite list used
    ///         to be read out of bounds inside the decoder (undefined
    ///         behaviour); now it fails loudly at the API boundary.
    static void validateSnapshot(const SimulationSnapshot& snap) {
        if (!snap.energyPerMinute.empty() &&
            snap.energyPerMinute.size() != snap.satellites.size()) {
            throw invalid_argument(
                "SimulationSnapshot.energyPerMinute must be empty (uniform "
                "1.0) or have exactly one entry per satellite: got " +
                to_string(snap.energyPerMinute.size()) + " entries for " +
                to_string(snap.satellites.size()) + " satellites.");
        }
    }

    vector<TerminalDef> registry_;
};