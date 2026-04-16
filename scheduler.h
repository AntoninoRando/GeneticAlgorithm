#pragma once

// =============================================================================
//  scheduler_gp.hpp  –  Genetic Programming scheduler for LEO satellites
//
//  ARCHITECTURE OVERVIEW
//  ─────────────────────
//  Each individual in the meta-GA is an *expression tree* (a GP tree) that
//  maps (Job, Satellite) pairs to a real-valued priority score.  The
//  simulation is fixed: every job has a pre-computed list of Opportunity
//  windows.  The scheduler derived from the tree works as follows:
//
//    1. For every unscheduled job j, evaluate tree(j, bestSatellite(j)) → score.
//    2. Sort jobs by descending score.
//    3. Greedily assign each job to its best-fitting opportunity that does not
//       conflict with already-assigned slots (respecting the minGap constraint).
//
//  ADAPTABILITY
//  ────────────
//  Terminals are registered at runtime via TerminalRegistry.  To add a new
//  field to Job or Satellite, simply add one line to buildTerminalRegistry().
//  The GP tree representation, operators, and GA machinery are completely
//  field-agnostic.
//
//  OPERATORS
//    +  -  *  /safe  MAX  MIN  NEG  ABS  INV
//
//  TERMINAL KINDS
//    JOB       – reads a double field of a Job
//    SATELLITE – reads a double field of the assigned Satellite
//    CONST     – a random ephemeral constant in [-10, 10]
// =============================================================================

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace std;


// ─────────────────────────────────────────────────────────────────────────────
//  DATA STRUCTS  (unchanged from original, fields may be extended freely)
// ─────────────────────────────────────────────────────────────────────────────

struct Opportunity {
    int satelliteId;
    int startMinute;
    int endMinute;
};

struct Job {
    int    id;
    string name;
    int    durationMinutes;
    int    dueMinute;
    double priority;
    vector<Opportunity> opportunities;

    time_t arrivalTime      = 0;
    double taskSize         = 0.0;
    double initialDeadline  = 0.0;
    double remainingDeadline = 0.0;
};

struct Satellite {
    int    id;
    string name;

    vector<Satellite> listeningDome;

    int    activeTasks          = 0;
    double RemainingEnergy      = 0.0;
    double ComputingLoad        = 0.0;
    double ComputingCapability  = 0.0;
};


// ─────────────────────────────────────────────────────────────────────────────
//  TERMINAL REGISTRY
//  Maps a human-readable name to a lambda (Job, Satellite) → double.
//  Add any new field here; nothing else needs to change.
// ─────────────────────────────────────────────────────────────────────────────

using FieldAccessor = function<double(const Job&, const Satellite&)>;

struct TerminalDef {
    string        name;
    FieldAccessor accessor;
};

static vector<TerminalDef> buildTerminalRegistry() {
    // ── Job fields ────────────────────────────────────────────────────────
    vector<TerminalDef> reg = {
        { "job.durationMinutes",   [](const Job& j, const Satellite&) { return static_cast<double>(j.durationMinutes);   } },
        { "job.dueMinute",         [](const Job& j, const Satellite&) { return static_cast<double>(j.dueMinute);         } },
        { "job.priority",          [](const Job& j, const Satellite&) { return j.priority;                               } },
        { "job.arrivalTime",       [](const Job& j, const Satellite&) { return static_cast<double>(j.arrivalTime);       } },
        { "job.taskSize",          [](const Job& j, const Satellite&) { return j.taskSize;                               } },
        { "job.initialDeadline",   [](const Job& j, const Satellite&) { return j.initialDeadline;                        } },
        { "job.remainingDeadline", [](const Job& j, const Satellite&) { return j.remainingDeadline;                      } },
        { "job.opportunityCount",  [](const Job& j, const Satellite&) { return static_cast<double>(j.opportunities.size()); } },
    // ── Satellite fields ──────────────────────────────────────────────────
        { "sat.activeTasks",         [](const Job&, const Satellite& s) { return static_cast<double>(s.activeTasks);       } },
        { "sat.RemainingEnergy",     [](const Job&, const Satellite& s) { return s.RemainingEnergy;                        } },
        { "sat.ComputingLoad",       [](const Job&, const Satellite& s) { return s.ComputingLoad;                          } },
        { "sat.ComputingCapability", [](const Job&, const Satellite& s) { return s.ComputingCapability;                    } },
    // ── Derived / cross-domain signals ────────────────────────────────────
        { "job.slack",             [](const Job& j, const Satellite&) {
            // slack = dueMinute – durationMinutes  (static urgency proxy)
            return static_cast<double>(j.dueMinute - j.durationMinutes);
        }},
        { "sat.loadRatio",         [](const Job&, const Satellite& s) {
            return (s.ComputingCapability > 0.0)
                ? s.ComputingLoad / s.ComputingCapability
                : 0.0;
        }},
    };
    // ── Adding a new field is ONE line here, e.g.:
    // reg.push_back({ "job.myNewField", [](const Job& j, const Satellite&){ return j.myNewField; } });
    return reg;
}


// ─────────────────────────────────────────────────────────────────────────────
//  EXPRESSION TREE
// ─────────────────────────────────────────────────────────────────────────────

enum class NodeType            { ADD, SUB, MUL, DIV, MAX_OP, MIN_OP, NEG, ABS, INV, TERMINAL, CONST };
static constexpr int ARITY[] = {   2,   2,   2,   2,      2,      2,   1,   1,   1,        0,     0 };




/// @brief A node in the GP expression tree.
struct ExprNode {
    #pragma region FIELDS ------------------------------------------------------
    NodeType nodeType;
    int terminalIndex = -1;  // For TERMINAL: index into TerminalRegistry.
    double constValue = 0.0; // For CONST: the ephemeral constant.
    vector<ExprNode> children;
    #pragma endregion ----------------------------------------------------------



    #pragma region CORE --------------------------------------------------------
    /// @brief Evaluates the subtree rooted here given a (job, satellite) context.
    /// @param job The job being scored.
    /// @param sat The satellite being considered for that job.
    /// @param registry The terminal registry.
    /// @return The computed priority score for this (job, satellite) pair.
    ///
    /// @throws runtime_error if the node type is invalid.
    double eval(const Job& job, const Satellite& sat,
                const vector<TerminalDef>& registry) const
    {
        switch (nodeType) {
            case NodeType::TERMINAL:
                return registry[terminalIndex].accessor(job, sat);
            case NodeType::CONST:
                return constValue;
            case NodeType::ADD:
                return children[0].eval(job, sat, registry) + children[1].eval(job, sat, registry);
            case NodeType::SUB:
                return children[0].eval(job, sat, registry) - children[1].eval(job, sat, registry);
            case NodeType::MUL:
                return children[0].eval(job, sat, registry) * children[1].eval(job, sat, registry);
            case NodeType::DIV: {
                double denom = children[1].eval(job, sat, registry);
                return (abs(denom) < 1e-9) ? 0.0 : children[0].eval(job, sat, registry) / denom;
            }
            case NodeType::MAX_OP:
                return max(children[0].eval(job, sat, registry), children[1].eval(job, sat, registry));
            case NodeType::MIN_OP:
                return min(children[0].eval(job, sat, registry), children[1].eval(job, sat, registry));
            case NodeType::NEG:
                return -children[0].eval(job, sat, registry);
            case NodeType::ABS:
                return abs(children[0].eval(job, sat, registry));
            case NodeType::INV: {
                double v = children[0].eval(job, sat, registry);
                return (abs(v) < 1e-9) ? 0.0 : 1.0 / v;
            }
        }

        throw runtime_error("Invalid node type" + to_string(static_cast<int>(nodeType)) + " in ExprNode::eval");
    }
    #pragma endregion ----------------------------------------------------------



    #pragma region UTILITIES ---------------------------------------------------
    // Count nodes in the subtree.
    int size() const {
        int s = 1;
        for (const auto& c : children) s += c.size();
        return s;
    }

    /// @brief Pretty-print the expression as a string.
    string toString(const vector<TerminalDef>& registry) const {
        switch (nodeType) {
            case NodeType::TERMINAL: return registry[terminalIndex].name;
            case NodeType::CONST: {
                ostringstream oss;
                oss << fixed << setprecision(3) << constValue;
                return oss.str();
            }
            case NodeType::ADD:    return "(" + children[0].toString(registry) + " + "   + children[1].toString(registry) + ")";
            case NodeType::SUB:    return "(" + children[0].toString(registry) + " - "   + children[1].toString(registry) + ")";
            case NodeType::MUL:    return "(" + children[0].toString(registry) + " * "   + children[1].toString(registry) + ")";
            case NodeType::DIV:    return "(" + children[0].toString(registry) + " /? "  + children[1].toString(registry) + ")";
            case NodeType::MAX_OP: return "MAX(" + children[0].toString(registry) + ", " + children[1].toString(registry) + ")";
            case NodeType::MIN_OP: return "MIN(" + children[0].toString(registry) + ", " + children[1].toString(registry) + ")";
            case NodeType::NEG:    return "(-" + children[0].toString(registry) + ")";
            case NodeType::ABS:    return "ABS(" + children[0].toString(registry) + ")";
            case NodeType::INV:    return "(1/" + children[0].toString(registry) + ")";
        }
        return "?";
    }

    // Collect pointers to all nodes by index (pre-order), for subtree ops.
    void collectNodes(vector<ExprNode*>& out) {
        out.push_back(this);
        for (auto& c : children) c.collectNodes(out);
    }
    void collectNodes(vector<const ExprNode*>& out) const {
        out.push_back(this);
        for (const auto& c : children) c.collectNodes(out);
    }
    #pragma endregion ----------------------------------------------------------
};


// ─────────────────────────────────────────────────────────────────────────────
//  TREE BUILDER  (grow / full / ramped-half-and-half)
// ─────────────────────────────────────────────────────────────────────────────

class TreeBuilder {
public:
    explicit TreeBuilder(const vector<TerminalDef>& registry, mt19937& rng)
        : registry_(registry), rng_(rng)
    {
        // Binary operators
        binaryOps_ = { NodeType::ADD, NodeType::SUB, NodeType::MUL,
                       NodeType::DIV, NodeType::MAX_OP, NodeType::MIN_OP };
        // Unary operators
        unaryOps_  = { NodeType::NEG, NodeType::ABS, NodeType::INV };
    }

    // Grow method: at each internal node, randomly pick operator or terminal.
    ExprNode grow(int maxDepth) {
        if (maxDepth == 0 || (maxDepth < 3 && randomReal() < 0.55)) {
            return makeTerminal();
        }
        return makeOperatorNode(maxDepth, /*full=*/false);
    }

    // Full method: every leaf is at exactly maxDepth.
    ExprNode full(int maxDepth) {
        if (maxDepth == 0) return makeTerminal();
        return makeOperatorNode(maxDepth, /*full=*/true);
    }

    // Ramped half-and-half population.
    vector<ExprNode> rampedHalfAndHalf(int count, int minDepth, int maxDepth) {
        vector<ExprNode> pop;
        pop.reserve(count);
        for (int i = 0; i < count; ++i) {
            int depth = minDepth + (i % (maxDepth - minDepth + 1));
            if (i % 2 == 0) pop.push_back(grow(depth));
            else             pop.push_back(full(depth));
        }
        return pop;
    }

private:
    const vector<TerminalDef>& registry_;
    mt19937& rng_;
    vector<NodeType> binaryOps_;
    vector<NodeType> unaryOps_;

    double randomReal() {
        uniform_real_distribution<double> d{0.0, 1.0};
        return d(rng_);
    }

    int randomInt(int lo, int hi) {
        uniform_int_distribution<int> d{lo, hi};
        return d(rng_);
    }

    ExprNode makeTerminal() {
        ExprNode node;
        // 20% chance of ephemeral constant.
        if (randomReal() < 0.20) {
            node.nodeType   = NodeType::CONST;
            node.constValue = (randomReal() * 20.0) - 10.0;
        } else {
            node.nodeType      = NodeType::TERMINAL;
            node.terminalIndex = randomInt(0, static_cast<int>(registry_.size()) - 1);
        }
        return node;
    }

    ExprNode makeOperatorNode(int maxDepth, bool forceOperator) {
        ExprNode node;

        // Choose arity: binary ~70%, unary ~30%.
        bool binary = (randomReal() < 0.70);
        if (binary) {
            node.nodeType = binaryOps_[randomInt(0, static_cast<int>(binaryOps_.size()) - 1)];
            if (forceOperator) {
                node.children.push_back(full(maxDepth - 1));
                node.children.push_back(full(maxDepth - 1));
            } else {
                node.children.push_back(grow(maxDepth - 1));
                node.children.push_back(grow(maxDepth - 1));
            }
        } else {
            node.nodeType = unaryOps_[randomInt(0, static_cast<int>(unaryOps_.size()) - 1)];
            if (forceOperator) node.children.push_back(full(maxDepth - 1));
            else               node.children.push_back(grow(maxDepth - 1));
        }
        return node;
    }
};


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

    GreedyDecoder(const vector<Satellite>& satellites, const vector<Job>& jobs,
                  const vector<TerminalDef>& registry)
        : satellites_(satellites), jobs_(jobs), registry_(registry) {}

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


// ─────────────────────────────────────────────────────────────────────────────
//  INDIVIDUAL (GP individual)
// ─────────────────────────────────────────────────────────────────────────────

struct GPIndividual {
    ExprNode       tree;
    ScheduleResult result;

    bool operator<(const GPIndividual& o) const {
        return result.fitness < o.result.fitness;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
//  GENETIC OPERATORS ON TREES
// ─────────────────────────────────────────────────────────────────────────────

class TreeOperators {
public:
    explicit TreeOperators(mt19937& rng, TreeBuilder& builder, int maxDepth = 7)
        : rng_(rng), builder_(builder), maxTreeDepth_(maxDepth) {}

    // ── Subtree crossover ────────────────────────────────────────────────────
    // Build two new trees by substituting random subtrees.
    // Uses index-based helpers to avoid raw pointer invalidation.

    static ExprNode getSubtreeAt(const ExprNode& node, int target, int& cur) {
        if (cur == target) return node;
        for (const auto& c : node.children) {
            ++cur;
            ExprNode r = getSubtreeAt(c, target, cur);
            if (cur >= target) return r;
        }
        return node;
    }
    static ExprNode getSubtree(const ExprNode& root, int idx) {
        int cur = 0; return getSubtreeAt(root, idx, cur);
    }

    static bool setSubtreeAt(ExprNode& node, int target, const ExprNode& repl, int& cur) {
        if (cur == target) { node = repl; return true; }
        for (auto& c : node.children) {
            ++cur;
            if (setSubtreeAt(c, target, repl, cur)) return true;
        }
        return false;
    }
    static ExprNode setSubtree(ExprNode root, int idx, const ExprNode& repl) {
        int cur = 0; setSubtreeAt(root, idx, repl, cur); return root;
    }

    pair<ExprNode, ExprNode> crossover(const ExprNode& a, const ExprNode& b) {
        int idxA = randomInt(0, a.size() - 1);
        int idxB = randomInt(0, b.size() - 1);

        ExprNode subA = getSubtree(a, idxA);
        ExprNode subB = getSubtree(b, idxB);

        ExprNode childA = setSubtree(a, idxA, subB);
        ExprNode childB = setSubtree(b, idxB, subA);

        if (childA.size() > maxTreeSize()) childA = builder_.grow(maxTreeDepth_);
        if (childB.size() > maxTreeSize()) childB = builder_.grow(maxTreeDepth_);

        return {childA, childB};
    }

    // ── Point mutation ───────────────────────────────────────────────────────
    // Recursively walk and return a mutated copy.  This avoids UAF from raw
    // pointer lists becoming stale after a child vector reallocation.
    ExprNode mutateNode(ExprNode node, double mutationRate) {
        if (randomReal() < mutationRate) {
            switch (node.nodeType) {
                case NodeType::TERMINAL:
                    if (randomReal() < 0.15) {
                        node.nodeType   = NodeType::CONST;
                        node.constValue = (randomReal() * 20.0) - 10.0;
                        node.children.clear();
                    } else {
                        node.terminalIndex = randomInt(0, terminalCount_ - 1);
                    }
                    return node;
                case NodeType::CONST: {
                    normal_distribution<double> gauss{0.0, 1.0};
                    node.constValue += gauss(rng_);
                    return node;
                }
                default:
                    if (randomReal() < 0.10)
                        return builder_.grow(3);
                    break;
            }
        }
        for (auto& child : node.children)
            child = mutateNode(move(child), mutationRate);
        return node;
    }

    void mutate(ExprNode& tree, double mutationRate) {
        tree = mutateNode(move(tree), mutationRate);
        if (tree.size() > maxTreeSize()) tree = builder_.grow(maxTreeDepth_);
    }

    // ── Hoist mutation ───────────────────────────────────────────────────────
    // Replace the whole tree with a copy of one of its non-root subtrees.
    void collectSubtrees(const ExprNode& node, vector<ExprNode>& out, bool isRoot) {
        if (!isRoot) out.push_back(node);
        for (const auto& c : node.children) collectSubtrees(c, out, false);
    }

    void hoist(ExprNode& tree) {
        vector<ExprNode> subtrees;
        collectSubtrees(tree, subtrees, true);
        if (subtrees.empty()) return;
        tree = move(subtrees[randomInt(0, static_cast<int>(subtrees.size()) - 1)]);
    }

    void setTerminalCount(int n) { terminalCount_ = n; }

private:
    mt19937&     rng_;
    TreeBuilder& builder_;
    int          maxTreeDepth_;
    int          terminalCount_ = 1;

    int maxTreeSize() const { return 1 << (maxTreeDepth_ + 1); }

    double randomReal() {
        uniform_real_distribution<double> d{0.0, 1.0};
        return d(rng_);
    }
    int randomInt(int lo, int hi) {
        uniform_int_distribution<int> d{lo, hi};
        return d(rng_);
    }
};


// ─────────────────────────────────────────────────────────────────────────────
//  SCHEDULER GP  – the meta-GA
// ─────────────────────────────────────────────────────────────────────────────

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


// ─────────────────────────────────────────────────────────────────────────────
//  DATA FACTORIES  (same as original, unchanged)
// ─────────────────────────────────────────────────────────────────────────────

static vector<Satellite> createSatellites() {
    return {
        {0, "SEO-A", {{1, "SEO-B", {}, 0, 0.0, 0.0, 0.0}, {2, "SEO-C", {}, 0, 0.0, 0.0, 0.0}}, 0, 100.0, 0.20, 18.0},
        {1, "SEO-B", {{0, "SEO-A", {}, 0, 0.0, 0.0, 0.0}, {2, "SEO-C", {}, 0, 0.0, 0.0, 0.0}}, 0,  98.0, 0.25, 16.5},
        {2, "SEO-C", {{0, "SEO-A", {}, 0, 0.0, 0.0, 0.0}, {1, "SEO-B", {}, 0, 0.0, 0.0, 0.0}}, 0,  95.0, 0.18, 19.0}
    };
}

static vector<Job> createJobs() {
    const time_t epoch = 1700000000;
    return {
        {0, "Target-01",  8, 100, 5.0, {{0,  30,  38}, {1,  42,  50}, {2,  55,  63}}, epoch + 10*60, 10.0, 100.0, 100.0},
        {1, "Target-02", 10, 145, 7.0, {{0,  70,  80}, {1,  92, 102}, {2, 110, 120}}, epoch + 55*60, 14.0, 145.0, 145.0},
        {2, "Target-03", 12, 190, 9.0, {{0, 125, 137}, {1, 140, 152}, {2, 162, 174}}, epoch+105*60, 17.0, 190.0, 190.0},
        {3, "Target-04",  9, 205, 4.0, {{0, 155, 164}, {1, 175, 184}, {2, 188, 197}}, epoch+138*60,  9.5, 205.0, 205.0},
        {4, "Target-05",  7, 230, 6.5, {{0, 180, 187}, {1, 198, 205}, {2, 216, 223}}, epoch+165*60, 11.0, 230.0, 230.0},
        {5, "Target-06", 11, 280, 8.5, {{0, 220, 231}, {1, 240, 251}, {2, 255, 266}}, epoch+205*60, 16.0, 280.0, 280.0},
        {6, "Target-07",  8, 315, 5.5, {{0, 260, 268}, {1, 272, 280}, {2, 295, 303}}, epoch+245*60, 10.5, 315.0, 315.0},
        {7, "Target-08", 10, 350, 7.5, {{0, 300, 310}, {1, 320, 330}, {2, 338, 348}}, epoch+285*60, 13.5, 350.0, 350.0},
        {8, "Target-09", 12, 390, 9.5, {{0, 335, 347}, {1, 355, 367}, {2, 370, 382}}, epoch+320*60, 18.0, 390.0, 390.0},
        {9, "Target-10",  9, 430, 6.0, {{0, 365, 374}, {1, 385, 394}, {2, 405, 414}}, epoch+350*60, 12.0, 430.0, 430.0}
    };
}


// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────

int gp_main() {
    auto satellites = createSatellites();
    auto jobs       = createJobs();

    cout << "=== LEO Satellite Scheduler — Genetic Programming ===\n";
    cout << "Terminals: ";
    auto reg = buildTerminalRegistry();
    for (size_t i = 0; i < reg.size(); ++i)
        cout << reg[i].name << (i + 1 < reg.size() ? ", " : "\n");
    cout << "Operators: +, -, *, /safe, MAX, MIN, NEG, ABS, INV\n\n";

    SchedulerGP gp(satellites, jobs);

    GPIndividual best = gp.solve(
        /*populationSize=*/ 200,
        /*generations=*/    300,
        /*crossoverRate=*/  0.80,
        /*mutationRate=*/   0.08
    );

    gp.printSchedule(best);
    return 0;
}