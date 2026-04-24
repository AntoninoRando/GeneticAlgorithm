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
    int            id;
    ExprNode       tree;

    /// @brief Amount of jobs successfully scheduled by this individual's tree 
    /// when used as a priority function during a simulation.
    double         fitness;

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
    SchedulerGP(vector<Satellite> satellites, vector<Job> jobs)
        : satellites_(move(satellites))
        , jobs_(move(jobs))
        , rng_(random_device{}())
        , registry_(buildTerminalRegistry())
        , builder_(registry_, rng_)
        , ops_(rng_, builder_, /*maxDepth=*/7)
    {
        ops_.setTerminalCount(static_cast<int>(registry_.size()));
    }
    #pragma endregion ----------------------------------------------------------

    
    
    #pragma region PUBLIC API --------------------------------------------------



    #pragma region CORE --------------------------------------------------------
    void initialize(int populationSize,
                    double crossoverRate, 
                    double mutationRate,
                    int elitismCount = 2)
    {
        this->populationSize = populationSize;
        this->crossoverRate  = crossoverRate;
        this->mutationRate   = mutationRate;
        this->elitismCount   = std::max(0, std::min(elitismCount, populationSize));

        gen = -1;
        
        // Initialize population with random expression trees.
        vector<ExprNode> seeds = builder_.rampedHalfAndHalf(populationSize, 2, 6);
        population.resize(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            population[i].tree   = move(seeds[i]);
            population[i].id     = i;
        }
    }

    void solveNextGeneration()
    {
        ++gen;

        // Sort current population by fitness (ascending).
        sort(population.begin(), population.end()); 

        // Prepare next generation.
        vector<GPIndividual> next;
        next.reserve(populationSize);
        
        // --- 1. Elitism: keep top individuals unchanged.
        for (int i = 0; i < elitismCount; ++i) {
            next.push_back(population[population.size() - 1 - i]);
        }

        // --- 2. Generate offspring until we have a full next generation.
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
            ops_.mutate(childB.tree, mutationRate);

            // Occasional hoist to fight bloat.
            if (randomReal() < 0.05) ops_.hoist(childA.tree);
            if (randomReal() < 0.05) ops_.hoist(childB.tree);

            next.push_back(move(childA));
            if (static_cast<int>(next.size()) < populationSize)
                next.push_back(move(childB));
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

        printRandomPopulationTrees(population);
    }

    const vector<GPIndividual>& getPopulation() const {
        return population;
    }

    vector<GPIndividual> topIndividuals(int k=1) const {
        vector<GPIndividual> topK = population;
        std::sort(topK.begin(), topK.end());
        if (k < static_cast<int>(topK.size())) {
            topK.resize(k);
        }
        return topK;
    }
    #pragma endregion ----------------------------------------------------------
    #pragma endregion ----------------------------------------------------------


private:
    #pragma region FIELDS ------------------------------------------------------
    vector<Satellite>    satellites_;
    vector<Job>          jobs_;
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
    int    elitismCount     = 2;
    
    /*
        Current state of the algorithm.
    */
    /// @brief The current generation number (0-indexed).
    int gen = 0;
    vector<GPIndividual> population;
    #pragma endregion ----------------------------------------------------------


    
    const GPIndividual& tournamentSelect(const vector<GPIndividual>& pop, int k) {
        int best = randomInt(0, static_cast<int>(pop.size()) - 1);
        for (int i = 1; i < k; ++i) {
            int cand = randomInt(0, static_cast<int>(pop.size()) - 1);
            if (pop[cand].fitness > pop[best].fitness) best = cand;
        }
        return pop[best];
    }



    #pragma region UTILITIES ---------------------------------------------------
    double randomReal() {
        uniform_real_distribution<double> d{0.0, 1.0};
        return d(rng_);
    }
    int randomInt(int lo, int hi) {
        std::uniform_int_distribution<int> d{lo, hi};
        return d(rng_);
    }

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