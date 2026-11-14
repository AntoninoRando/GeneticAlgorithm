#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "registry.h"
#include "tree_builder.h"
#include "tree_operators.h"

using namespace std;
// -----------------------------------------------------------------------------


/*
    This file defines the SchedulerGP class, which implements a genetic 
    programming algorithm to evolve expression trees that can be used as 
    priority functions for scheduling jobs on satellites.

    PUBLIC API:
    The GP algorithm must first be initialized with the set of satellites and 
    jobs via the `initialize()` method, which also sets the GP parameters such 
    as population size and mutation rate. This method also generates the initial
    population of random expression trees.

    The `solveNextGeneration()` method runs one iteration of the GP algorithm, 
    which includes selection, crossover, and mutation to produce the next 
    generation of candidate solutions.

    The `printSchedule()` method can be used to print the best individual's 
    fitness and expression tree, as well as a few random samples from the 
    population for insight into diversity. The printing of the schedule status
    every k generation is let to the caller to avoid unnecessary overhead.

    INDIVIDUAL FITNESS:
    The fitness of an individual is determined by the amount of jobs it manages
    to complete. That is, an expression tree which, when used as a priority 
    function in the greedy decoder, allowed to complete more jobs will be 
    considered more fit than an expression tree which, when used under the 
    same conditions of jobs and satellites, allowed to complete fewer jobs.

    The GP algorithm solely evolves the expression trees based on their fitness.
    The actual fitness value must be set externally by evaluating the 
    individual's tree during a simulation.

    LITERATURE NOTES  (keys refer to thesis/references.bib; section numbers are
    those of the cited papers, see thesis/literature_notes.md)
    ----------------------------------------------------------------------
    Paper                          bib key             what it is used for here
    -----------------------------  ------------------  ------------------------
    Koza 1994                      koza1994gp          GP loop, closure, depth cap
    Sun et al. 2024                mtgp_multicloud     loop, parameters, 3-tree ablation
    Zhang & Yang 2025 (MTGPER)     zhang2025mtgper     GP for SEC, duplicate test
    Xu et al. 2023                 gp_fog              multi-tree GP, breeding
    Yang et al. 2025 (DTGPAM)      dtgpam, yang2025thesis   adaptive terminal weighting
    Zhang et al. 2018              zhang2018mtgp       multi-tree / sMTGP origin
    Nguyen et al. 2017             nguyen2017gpsurvey  framework, multiple decisions
    Magliarisi et al. 2025, 2026   magliarisi2025orbit, icdcs_sc, magliarisi2026thesis
                                                       problem, constraints, metrics

    The generational loop below (rank, select parents by tournament, copy the
    elites, breed the rest, repeat) is the one described in koza1994gp §3 and
    used unchanged by mtgp_multicloud §4.1, zhang2025mtgper §3.2.1 and gp_fog
    §VI-A.  The deviations from the papers' settings are collected here so each
    can be checked against its source:

      parameter           this project     papers
      ------------------  ---------------  -----------------------------------
      population          100 (CLI)        200 mtgp_multicloud T.6, zhang2025mtgper
                                           T.3; 1024 dtgpam §V-C
      generations         200 (CLI)        50 mtgp_multicloud T.6; 51 zhang2025mtgper
                                           T.3, dtgpam §V-C
      elitism             2                5 mtgp_multicloud T.6; 1 dtgpam §V-C
                                           ("Hall of Fame" in zhang2025mtgper §3.2.1)
      tournament size     5                7 mtgp_multicloud, dtgpam; 2 zhang2025mtgper
      crossover / mut.    0.80 / 0.08/node 0.80 / 0.15 per individual (T.6, T.3);
                                           0.65 / 0.30 for DTGPAM (yang2025thesis T.3.7)
      max depth           7 (CLI 4-7)      8 mtgp_multicloud, dtgpam; 7 zhang2025mtgper
      initial depth       2..maxDepth      2-7 / 2-6 / 1-3 (see TreeBuilder)

    Selection pressure is controlled by the tournament size: a larger tournament
    converges faster and keeps less diversity.  The papers do not tune it (Sun
    and Yang copy 7 from earlier work, Zhang & Yang use a small 2); it is
    exposed as `tournamentSize` so that it can be studied.

    MEASURED EFFECT of the opt-in options on THIS problem
    (thesis/experiments/run_ablation.py: population 100, depth cap 4, 300
    generations, trained on worlds 0-19 and scored on the 10 held-out worlds;
    test completion ratio C, mean over 20 paired seeds, Mann-Whitney vs the
    unchanged baseline).  First set = seeds 0-19, second = fresh seeds 100-119
    run afterwards to confirm the first set's findings:

      option                    C seeds 0-19         C seeds 100-119
      ------------------------  -------------------  -------------------
      baseline (all off)        0.223                0.227
      tournamentSize=7          0.224  (p=0.73)      -
      initialMaxDepth=3         0.235  (p=0.29)      -
      internalCrossoverBias=.9  0.226  (p=0.96)      -
      cloneRetries=3            0.251  (p=0.005)     0.246  (p=0.27)
      terminalWeighting         0.221  (p=0.68)      -
      prune constant terminals  0.221  (p=0.47)      -
      trackBusyUntil            0.232  (p=0.33)      -
      3 worlds per evaluation   0.248  (p=0.014)     0.247  (p=0.045)

    What the gain is.  The baseline is bimodal: in 16 of 40 runs (both sets) it
    ends on a degenerate FCFS-like rule such as `job.arrivalTime` (C about
    0.19), while the other runs reach about 0.24-0.26.  With cloneRetries=3
    0 of 40 runs fail this way, with 3 worlds 1 of 40 (Fisher p < 1e-4, but
    the 0.22 threshold was chosen after looking at the data).  The final
    structural diversity is 0.57 for the baseline against 0.97 and 0.86.  So
    the two options mostly remove a premature-convergence failure mode
    (the motivation of the repetition test in zhang2025mtgper §3.2.1); the
    typical successful run is only slightly better (median C 0.242 -> 0.249 /
    0.251).  Mean response time is slightly WORSE (14.3 -> 14.9 / 14.6 min, first set),
    as more jobs get scheduled, and later.  The options with no measurable
    effect are kept because they are cheap and documented, not because they
    helped.  The p-values are not corrected for the 10 comparisons.

    Independent runs.  The papers report 30 (mtgp_multicloud §5.2, zhang2025mtgper
    §4.1) or 50 (dtgpam §VI) independent runs per scenario and compare them with
    Wilcoxon / Friedman tests.  That is only meaningful if each run is
    reproducible and independent, which is why the evolution takes an explicit
    seed and every offspring pair has its own RNG stream (see
    solveNextGeneration()).
*/



#pragma region TYPES -----------------------------------------------------------
/// @brief Represents an individual in the GP population, consisting of an
/// expression tree and its evaluated fitness result.
///
/// The `<` operator is defined to allow sorting individuals by fitness in 
/// ascending order.
struct GPIndividual {
    /// @brief Unique identifier, assigned at initialisation and for every
    /// offspring.  Defaulted so a freshly constructed individual never
    /// exposes an indeterminate value (reading one is UB).
    int            id = -1;
    ExprNode       tree;

    /// @brief Scalar fitness written back by FitnessEvaluator::evaluate()
    /// (weighted composite of completion ratio, mean response time and
    /// energy).  Defaults to 0.0 so offspring that have not been evaluated
    /// yet hold a defined value instead of uninitialised memory.
    double         fitness = 0.0;

    /// @brief Comparison operator for sorting individuals by fitness 
    /// (ascending).
    bool operator<(const GPIndividual& o) const {
        return fitness < o.fitness;
    }
};
#pragma endregion --------------------------------------------------------------



#pragma region ALGORITHM -------------------------------------------------------
class SchedulerGP {
public:
    static constexpr int kRandomTreesToPrint = 3;



    #pragma region CONSTRUCTORS ------------------------------------------------
    /// @param satellites / jobs  Accepted for API compatibility but currently
    ///        unused: the GP only evolves trees, and the world state reaches
    ///        the fitness evaluation through SimulationSnapshot.  (They were
    ///        previously copied into never-read members.)
    /// @param seed  < 0 (default): seed from std::random_device (fresh run
    ///        every time).  >= 0: fully deterministic, reproducible runs —
    ///        combined with the per-pair RNG streams in solveNextGeneration()
    ///        the same seed yields the same evolution regardless of thread
    ///        count or OpenMP scheduling.
    SchedulerGP(const vector<Satellite>& satellites, const vector<Job>& jobs,
                long long seed = -1)
        : rng_(seedFrom(seed))
        , registry_(buildTerminalRegistry())
        , builder_(registry_, rng_)
        , ops_(rng_, builder_, /*maxDepth=*/7)
    {
        (void)satellites;
        (void)jobs;
        ops_.setTerminalCount(static_cast<int>(registry_.size()));
    }
    #pragma endregion ----------------------------------------------------------

    
    
    #pragma region PUBLIC API --------------------------------------------------



    #pragma region CORE --------------------------------------------------------
    /// @param tournamentSize  Candidates per tournament (>= 1, clamped to the
    ///        population).  Default 5 reproduces the original behaviour.
    /// @param initialMaxDepth Maximum depth of the *initial* trees; -1 (default)
    ///        means "= maxDepth", the original behaviour.  The papers start
    ///        below the cap so crossover can grow trees (see
    ///        TreeBuilder::rampedHalfAndHalf for the values).
    /// @param internalCrossoverBias  Probability that a crossover point is an
    ///        operator node; 0 (default) = uniform over all nodes.  Koza uses
    ///        0.9 (koza1994gp §4).
    /// @param cloneRetries  Re-mutations tried for an offspring that is
    ///        structurally identical to a parent or an elite; 0 (default)
    ///        disables the check.  Structural stand-in for the behavioural
    ///        repetition test of zhang2025mtgper §3.2.1.  The one option with
    ///        a clear measured benefit (3 is a good value): it removed the
    ///        premature convergence that hit 16 of 40 baseline runs; see
    ///        "MEASURED EFFECT" at the top of this file.
    /// @param terminalWeighting  Adaptive terminal weighting for mutation
    ///        (Yang et al. 2025, dtgpam §IV-D / yang2025thesis §3.2.4).
    ///        Default false.
    /// @param depthAwareMutation  Grow the replacement subtree of subtree
    ///        mutation within the depth left below the cap, so that a
    ///        mutation is never discarded for exceeding it.  Default false.
    /// @param depthAwareCrossover  Draw the second crossover point among the
    ///        nodes for which both children respect the depth cap, so that
    ///        crossover never falls back to a copy of a parent.  Default
    ///        false.  Both follow the remedy of poli2008fieldguide §11.3.2
    ///        for depth limits; see the LITERATURE NOTES of tree_operators.h.
    /// @param excludedTerminals  Registry indices that are never drawn (for
    ///        example the constant terminals reported by
    ///        FitnessEvaluator::constantTerminals).  Empty = use them all.
    void initialize(int populationSize,
                    double crossoverRate, 
                    double mutationRate,
                    int elitismCount = 2,
                    int maxDepth     = 7,
                    double hoistRate = 0.05,
                    int tournamentSize = 5,
                    int initialMaxDepth = -1,
                    double internalCrossoverBias = 0.0,
                    int cloneRetries = 0,
                    bool terminalWeighting = false,
                    vector<int> excludedTerminals = {},
                    bool depthAwareMutation = false,
                    bool depthAwareCrossover = false)
    {
        this->populationSize = populationSize;
        this->crossoverRate  = crossoverRate;
        this->mutationRate   = mutationRate;
        this->elitismCount   = std::max(0, std::min(elitismCount, populationSize));
        this->maxDepth_      = std::max(1, maxDepth);
        this->hoistRate_     = std::max(0.0, std::min(1.0, hoistRate));
        this->tournamentSize_ = std::max(1, std::min(tournamentSize, std::max(1, populationSize)));
        this->cloneRetries_  = std::max(0, cloneRetries);
        this->terminalWeighting_ = terminalWeighting;
        this->internalCrossoverBias_ = std::max(0.0, std::min(1.0, internalCrossoverBias));
        ops_.setMaxDepth(this->maxDepth_);
        ops_.setInternalCrossoverBias(this->internalCrossoverBias_);
        this->depthAwareMutation_  = depthAwareMutation;
        this->depthAwareCrossover_ = depthAwareCrossover;
        ops_.setDepthAwareMutation(depthAwareMutation);
        ops_.setDepthAwareCrossover(depthAwareCrossover);

        // Active terminal set = registry minus the excluded indices.
        const int nTerm = static_cast<int>(registry_.size());
        vector<uint8_t> excluded(nTerm, 0);
        for (int t : excludedTerminals) {
            if (t < 0 || t >= nTerm)
                throw invalid_argument("excludedTerminals: index " + to_string(t) +
                                       " is outside the terminal registry (size " +
                                       to_string(nTerm) + ").");
            excluded[t] = 1;
        }
        activeTerminals_.clear();
        if (!excludedTerminals.empty()) {
            for (int t = 0; t < nTerm; ++t)
                if (!excluded[t]) activeTerminals_.push_back(t);
            if (activeTerminals_.empty())
                throw invalid_argument("excludedTerminals removes every terminal.");
        }
        builder_.setActiveTerminals(&activeTerminals_);
        terminalCdfActive_ = false;

        gen = -1;
        
        // Initialize population with random expression trees (ramped
        // half-and-half: koza1994gp §4, mtgp_multicloud §4.1, zhang2025mtgper
        // §3.2.1).  The minimum depth 2 is clamped so that a cap of 1 cannot
        // make the ramp's modulus zero (that used to divide by zero).
        const int initMax = (initialMaxDepth < 0)
                                ? this->maxDepth_
                                : std::max(1, std::min(initialMaxDepth, this->maxDepth_));
        vector<ExprNode> seeds = builder_.rampedHalfAndHalf(populationSize, std::min(2, initMax), initMax);
        population.resize(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            population[i].tree   = move(seeds[i]);
            population[i].id     = i;
        }
    }

    void solveNextGeneration()
    {
        ++gen;

        // Sort current population by fitness (ascending).  stable_sort so
        // that fitness ties keep a deterministic order — with plain sort the
        // tie order (and therefore tournament selection) varied from run to
        // run even with a fixed seed.
        stable_sort(population.begin(), population.end());

        const int popSize = static_cast<int>(population.size());

        // Adaptive terminal weighting (opt-in): learn, from the best
        // individuals of this generation, which terminals mutation should draw
        // more often.  Pure arithmetic on the sorted population; draws no
        // random numbers, so it cannot disturb the seeded RNG streams.
        updateTerminalDistribution();

        // Prepare the next generation with a *fixed* layout so it can be filled
        // concurrently: slots [0, elitismCount) hold the elites, and every
        // remaining slot is produced independently by the parallel offspring
        // loop below.  Pre-sizing (instead of push_back) is what makes the
        // write targets disjoint and therefore thread-safe.
        vector<GPIndividual> next(popSize);

        // --- 1. Elitism: keep top individuals unchanged.
        for (int i = 0; i < elitismCount; ++i) {
            next[i] = population[popSize - 1 - i];
        }

        // --- 2. Generate offspring in parallel.
        // One crossover yields two children, so we iterate over child *pairs*;
        // pair p writes to the two distinct, pre-allocated slots 2p and 2p+1.
        // The only shared state read inside the loop is `population` (read-only
        // during selection/copy) and `next` (each iteration writes its own two
        // slots), so the loop is data-race free.
        const int firstChild = elitismCount;
        const int nChildren  = popSize - elitismCount;   // may be 0
        const int nPairs     = (nChildren + 1) / 2;

        // Offspring ids: unique, deterministic, and never colliding with any
        // id from an earlier generation (elites keep their old, smaller ids).
        // Previously offspring ids were left uninitialised — reading
        // Individual.id from Python returned indeterminate memory.
        const int idBase = (gen + 1) * popSize;

        // Derive a base seed from the master RNG (consumed once, serially) so
        // that successive generations keep producing fresh offspring streams.
        const unsigned baseSeed = rng_();

        // Runtimes vary per pair (tree sizes differ), so dynamic scheduling
        // keeps the cores evenly loaded.  Each pair owns its own RNG stream
        // seeded from (baseSeed, p) — mt19937 is not thread-safe, and per-PAIR
        // (rather than per-thread) streams make the result independent of the
        // thread count and of how OpenMP happens to assign pairs to threads,
        // so seeded runs are bit-reproducible.  The registry is read-only and
        // safely shared; every iteration writes only its own two slots.
        #pragma omp parallel for schedule(dynamic)
        for (int p = 0; p < nPairs; ++p) {
            seed_seq      seq{baseSeed, static_cast<unsigned>(p)};
            mt19937       pairRng(seq);
            TreeBuilder   pairBuilder(registry_, pairRng);
            TreeOperators pairOps(pairRng, pairBuilder, maxDepth_);
            pairOps.setTerminalCount(static_cast<int>(registry_.size()));
            pairOps.setInternalCrossoverBias(internalCrossoverBias_);
            pairOps.setDepthAwareMutation(depthAwareMutation_);
            pairOps.setDepthAwareCrossover(depthAwareCrossover_);
            // Terminal restrictions apply to everything this pair builds or
            // mutates; both pointers are read-only here (written once, before
            // the parallel region), so sharing them is race-free.
            pairBuilder.setActiveTerminals(&activeTerminals_);
            pairBuilder.setTerminalDistribution(terminalCdfActive_ ? &terminalCdf_ : nullptr);

            uniform_real_distribution<double> unit(0.0, 1.0);

            const int slotA = firstChild + 2 * p;
            const int slotB = slotA + 1;

            // Tournament selection (koza1994gp used fitness-proportionate
            // selection; the scheduling papers all use tournaments:
            // mtgp_multicloud §4.1, zhang2025mtgper §3.2.1, gp_fog §VI-A.4).
            const GPIndividual& pA = tournamentSelectRng(population, tournamentSize_, pairRng);
            const GPIndividual& pB = tournamentSelectRng(population, tournamentSize_, pairRng);

            GPIndividual childA, childB;

            if (unit(pairRng) < crossoverRate) {
                auto [tA, tB] = pairOps.crossover(pA.tree, pB.tree);
                childA.tree   = move(tA);
                childB.tree   = move(tB);
            } else {
                childA.tree = pA.tree;
                childB.tree = pB.tree;
            }

            // Mutation
            pairOps.mutate(childA.tree, mutationRate);
            pairOps.mutate(childB.tree, mutationRate);

            // Occasional hoist to fight bloat.  The papers rely on the depth
            // cap alone (see tree_operators.h); hoist is this project's
            // addition.
            if (unit(pairRng) < hoistRate_) pairOps.hoist(childA.tree);
            if (unit(pairRng) < hoistRate_) pairOps.hoist(childB.tree);

            // Clone check (opt-in).  An offspring identical to a parent or to
            // an elite adds nothing to the population: it costs a full
            // decode and one more copy of an already represented behaviour.
            // zhang2025mtgper §3.2.1 handles this with a *behavioural*
            // repetition test in the first 30 of 51 generations, re-mutating
            // individuals that rank servers and tasks identically.  Here the
            // test is structural (cheap and independent of the world) and runs
            // in every generation; it is a conservative proxy, as two
            // different trees can still behave identically.  Retrying with a
            // per-node rate of at least 1/size makes at least one change
            // likely on each attempt.
            if (cloneRetries_ > 0) {
                for (GPIndividual* child : {&childA, &childB}) {
                    for (int r = 0; r < cloneRetries_ && isCloneOf(child->tree, pA, pB, next); ++r) {
                        const double boosted = std::max(mutationRate,
                            1.0 / std::max(1, child->tree.size()));
                        pairOps.mutate(child->tree, boosted);
                    }
                }
            }

            childA.id = idBase + slotA;
            childB.id = idBase + slotB;

            next[slotA] = move(childA);
            if (slotB < popSize) next[slotB] = move(childB);
        }

        population = move(next);
    }
    #pragma endregion ----------------------------------------------------------



    #pragma region INSIGHT -----------------------------------------------------
    void printSchedule() const {
        auto best = *max_element(population.begin(), population.end());
        cout << "Gen " << setw(4) << gen + 1
             << " | Best fitness: "  << fixed << setprecision(2) << best.fitness
             << " | Best tree: " << best.tree.toString(registry_) << "\n";

        // printRandomPopulationTrees(population);
    }

    vector<GPIndividual>& getPopulation() {
        return population;
    }

    /// @brief Fraction of the population that is structurally distinct
    /// (1.0 = every tree different, 1/N = all clones).  A cheap diversity
    /// indicator for comparing settings; the papers compare diversity
    /// indirectly through convergence curves (mtgp_multicloud Fig. 6,
    /// dtgpam §VI).
    double structuralDiversity() const {
        if (population.empty()) return 0.0;
        set<string> distinct;
        for (const auto& ind : population) distinct.insert(ind.tree.toString(registry_));
        return static_cast<double>(distinct.size()) / static_cast<double>(population.size());
    }

    /// @brief Maximum-likelihood terminal probabilities (Yang et al. 2025,
    /// dtgpam §IV-D; yang2025thesis §3.2.4, Algorithm 3, Eqs. 3.9-3.20).
    ///
    /// Given the terminal counts x_i observed in the best trees and a sampling
    /// ratio r, mutation draws terminal i with probability
    ///        P_i = r * p_i + (1 - r) / n
    /// (a mix of "learned" and uniform draws, Algorithm 1).  The p_i that make
    /// the observed counts most likely solve
    ///        max  sum_i x_i * ln(r * p_i + (1 - r) / n)
    ///        s.t. p_i >= 0, sum_i p_i = 1,
    /// a convex problem whose KKT conditions give
    ///        p_i = max(0, x_i / nu - (1 - r) / (n * r)),
    /// with nu fixed by sum_i p_i = 1.  nu is found by *water filling* in
    /// O(n log n) (yang2025thesis, Eq. 3.20 and Fig. 3.4): sort x
    /// descending, and take the largest set of k terminals for which
    /// nu_k = (x_1 + ... + x_k) / (1 + k c), c = (1 - r) / (n * r), keeps every
    /// chosen x_i / nu_k above c and every other one below it.
    /// Worked example from the thesis: x = {3, 5, 8}, r = 1/2  ->
    /// p = {1/24, 7/24, 2/3} (checked in tests/test_paper_features.py).
    /// Returns the uniform distribution when no terminal was observed.
    static vector<double> solveTerminalWeights(const vector<double>& x, double r) {
        const size_t n = x.size();
        if (n == 0) return {};
        const double total = std::accumulate(x.begin(), x.end(), 0.0);
        if (!(r > 0.0) || total <= 0.0)
            return vector<double>(n, 1.0 / static_cast<double>(n));
        if (r > 1.0) r = 1.0;

        const double c = (1.0 - r) / (static_cast<double>(n) * r);   // = phi / r

        vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return x[a] > x[b]; });

        // Find the number of "active" terminals k (water level).
        double prefix = 0.0;
        double nu = total;               // fallback: all terminals active
        for (size_t k = 1; k <= n; ++k) {
            prefix += x[order[k - 1]];
            if (prefix <= 0.0) break;
            const double nuK = prefix / (1.0 + static_cast<double>(k) * c);
            const bool kthActive = x[order[k - 1]] > nuK * c;
            const bool nextInactive = (k == n) || (x[order[k]] <= nuK * c);
            if (kthActive) nu = nuK;     // keeps the largest valid k if rounding blocks the exact one
            if (kthActive && nextInactive) break;
        }

        vector<double> p(n, 0.0);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            p[i] = std::max(0.0, x[i] / nu - c);
            sum += p[i];
        }
        if (sum <= 0.0) return vector<double>(n, 1.0 / static_cast<double>(n));
        for (auto& v : p) v /= sum;      // remove floating-point drift
        return p;
    }

    /// @brief Sampling ratio r for generation g: the "log rev" schedule
    /// r = 1 - 1 / (1 + 2 ln(g + 1)), which starts at 0 (pure uniform draws)
    /// and grows towards 1.  yang2025thesis §3.4.1 (Table 3.9; dtgpam §VI-B)
    /// compared six schedules and found the increasing ones best, log rev the
    /// best of them, hence "explore first, exploit later".  Capped at
    /// `kMaxSamplingRatio`: DTGPAM runs 51 generations, this project runs
    /// thousands, and at r = 1 a terminal absent from the elite could never be
    /// drawn again.  (The thesis does not state the log base; natural log is
    /// assumed.)
    static double terminalSamplingRatio(int generation) {
        const double r = 1.0 - 1.0 / (1.0 + 2.0 * std::log(static_cast<double>(generation) + 1.0));
        return std::max(0.0, std::min(kMaxSamplingRatio, r));
    }

    static constexpr double kMaxSamplingRatio = 0.9;

    /// @brief Returns copies of the best k individuals, best first.
    /// (The previous version sorted ascending and kept the *first* k
    /// elements, i.e. it returned the WORST k individuals.)
    vector<GPIndividual> topIndividuals(int k=1) const {
        vector<GPIndividual> topK = population;
        stable_sort(topK.begin(), topK.end(),
                    [](const GPIndividual& a, const GPIndividual& b) { return b < a; });
        if (k < static_cast<int>(topK.size())) {
            topK.resize(k);
        }
        return topK;
    }
    #pragma endregion ----------------------------------------------------------
    #pragma endregion ----------------------------------------------------------


private:
    #pragma region FIELDS ------------------------------------------------------
    mutable mt19937      rng_;
    vector<TerminalDef>  registry_;
    TreeBuilder          builder_;
    TreeOperators        ops_;

    /*
        Genetic Algorithm parameters.
    */
    int    populationSize = 200;
    double crossoverRate  = 0.80;
    double mutationRate   = 0.08;
    /// @brief Number of top individuals to carry over unchanged to next 
    /// generation.
    int    elitismCount   = 2;
    /// @brief Maximum depth of expression trees. Controls tree size cap 
    /// (max nodes ≈ 2^(maxDepth+1)).
    int    maxDepth_      = 7;
    /// @brief Probability per offspring that a hoist mutation is applied, 
    /// replacing the whole tree with one of its own subtrees to fight bloat.
    double hoistRate_     = 0.05;
    /// @brief Candidates per tournament (selection pressure).
    int    tournamentSize_ = 5;
    /// @brief Re-mutation attempts for clone offspring (0 = check disabled).
    int    cloneRetries_  = 0;
    /// @brief Probability that a crossover point is an internal node.
    double internalCrossoverBias_ = 0.0;
    /// @brief Adaptive terminal weighting on/off.
    bool   terminalWeighting_ = false;
    /// @brief Depth-aware subtree mutation / crossover (see tree_operators.h).
    bool   depthAwareMutation_  = false;
    bool   depthAwareCrossover_ = false;
    /// @brief Terminals that may be drawn; empty = the whole registry.
    vector<int>    activeTerminals_;
    /// @brief Cumulative distribution over the active terminals, rebuilt once
    /// per generation when terminal weighting is on.
    vector<double> terminalCdf_;
    bool           terminalCdfActive_ = false;
    
    /*
        Current state of the algorithm.
    */
    /// @brief The current generation number (0-indexed).
    int gen = 0;
    vector<GPIndividual> population;
    #pragma endregion ----------------------------------------------------------


    
    /// @brief Seeds the master RNG: negative → non-deterministic
    /// (random_device), otherwise the given seed for reproducible runs.
    static unsigned seedFrom(long long seed) {
        return seed < 0 ? random_device{}() : static_cast<unsigned>(seed);
    }

    /// @brief True if `t` equals a parent's tree or an elite already placed in
    /// `next` (slots [0, elitismCount) are filled before breeding starts and
    /// never written again, so reading them from the parallel loop is safe).
    bool isCloneOf(const ExprNode& t, const GPIndividual& pA, const GPIndividual& pB,
                   const vector<GPIndividual>& next) const {
        if (ExprNode::structurallyEqual(t, pA.tree) ||
            ExprNode::structurallyEqual(t, pB.tree)) return true;
        for (int e = 0; e < elitismCount; ++e)
            if (ExprNode::structurallyEqual(t, next[e].tree)) return true;
        return false;
    }

    /// @brief Rebuilds `terminalCdf_` from the top individuals.  Called at the
    /// start of solveNextGeneration(), when `population` is sorted ascending
    /// by fitness (best last).
    ///
    /// DTGPAM computes its weights from "the best n_tw individuals"
    /// (yang2025thesis §3.2.4, Algorithm 3, line 1); the text does not give
    /// n_tw, so the best 10% (at least 2) is used here.  Counting is over the
    /// active terminals only.  Weights apply to mutation (replacement subtrees
    /// and terminal redraws), not to the initial population, as in
    /// Algorithm 1.
    void updateTerminalDistribution() {
        terminalCdfActive_ = false;
        if (!terminalWeighting_) return;

        const double r = terminalSamplingRatio(gen);
        if (r <= 1e-9) return;                       // generation 0: uniform

        const int nReg = static_cast<int>(registry_.size());
        vector<int> position(nReg, -1);              // registry index -> slot
        const int nAct = activeTerminals_.empty() ? nReg
                                                  : static_cast<int>(activeTerminals_.size());
        for (int k = 0; k < nAct; ++k)
            position[activeTerminals_.empty() ? k : activeTerminals_[k]] = k;

        const int popSize = static_cast<int>(population.size());
        const int topK = std::min(popSize, std::max(2, popSize / 10));
        vector<double> counts(nAct, 0.0);
        for (int i = 0; i < topK; ++i)
            countTerminals(population[popSize - 1 - i].tree, position, counts);

        const vector<double> p = solveTerminalWeights(counts, r);
        terminalCdf_.assign(nAct, 0.0);
        double acc = 0.0;
        for (int k = 0; k < nAct; ++k) {
            acc += r * p[k] + (1.0 - r) / static_cast<double>(nAct);
            terminalCdf_[k] = acc;
        }
        terminalCdf_.back() = 1.0;                   // guard against drift
        terminalCdfActive_ = true;
    }

    static void countTerminals(const ExprNode& n, const vector<int>& position,
                               vector<double>& counts) {
        if (n.nodeType == NodeType::TERMINAL) {
            const int k = (n.terminalIndex >= 0 && n.terminalIndex < static_cast<int>(position.size()))
                              ? position[n.terminalIndex] : -1;
            if (k >= 0) counts[k] += 1.0;
        }
        for (const auto& c : n.children) countTerminals(c, position, counts);
    }

    /// @brief Tournament selection driven by a caller-supplied RNG so it can run
    /// concurrently: each thread passes its own private mt19937. Functionally
    /// identical to tournamentSelect() but with no shared RNG state.
    static const GPIndividual& tournamentSelectRng(const vector<GPIndividual>& pop,
                                                   int k, mt19937& rng) {
        uniform_int_distribution<int> pick(0, static_cast<int>(pop.size()) - 1);
        int best = pick(rng);
        for (int i = 1; i < k; ++i) {
            int cand = pick(rng);
            if (pop[cand].fitness > pop[best].fitness) best = cand;
        }
        return pop[best];
    }

    #pragma region UTILITIES ---------------------------------------------------
    /// @brief Prints the summary of a few random trees from the population
    /// for insight into diversity.
    void printRandomPopulationTrees(const vector<GPIndividual>& population) const {
        if (population.empty()) return;

        vector<int> indices(population.size());
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            indices[i] = i;
        }
        shuffle(indices.begin(), indices.end(), rng_);

        const int treesToPrint = std::min(kRandomTreesToPrint, static_cast<int>(population.size()));
        for (int i = 0; i < treesToPrint; ++i) {
            const GPIndividual& individual = population[indices[i]];
            cout << "  Tree sample " << (i + 1)
                 << " | Fitness: " << fixed << setprecision(2) << individual.fitness
                 << " | Size: " << individual.tree.size()
                 << " | Expr: " << individual.tree.toString(registry_)
                 << "\n";
        }
    }
    #pragma endregion ----------------------------------------------------------
};
#pragma endregion --------------------------------------------------------------