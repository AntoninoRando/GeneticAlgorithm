#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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
    void initialize(int populationSize,
                    double crossoverRate, 
                    double mutationRate,
                    int elitismCount = 2,
                    int maxDepth     = 7,
                    double hoistRate = 0.05)
    {
        this->populationSize = populationSize;
        this->crossoverRate  = crossoverRate;
        this->mutationRate   = mutationRate;
        this->elitismCount   = std::max(0, std::min(elitismCount, populationSize));
        this->maxDepth_      = std::max(1, maxDepth);
        this->hoistRate_     = std::max(0.0, std::min(1.0, hoistRate));
        ops_.setMaxDepth(this->maxDepth_);

        gen = -1;
        
        // Initialize population with random expression trees.
        vector<ExprNode> seeds = builder_.rampedHalfAndHalf(populationSize, 2, this->maxDepth_);
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

            uniform_real_distribution<double> unit(0.0, 1.0);

            const int slotA = firstChild + 2 * p;
            const int slotB = slotA + 1;

            const GPIndividual& pA = tournamentSelectRng(population, 5, pairRng);
            const GPIndividual& pB = tournamentSelectRng(population, 5, pairRng);

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

            // Occasional hoist to fight bloat.
            if (unit(pairRng) < hoistRate_) pairOps.hoist(childA.tree);
            if (unit(pairRng) < hoistRate_) pairOps.hoist(childB.tree);

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