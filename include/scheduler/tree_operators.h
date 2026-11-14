#pragma once

#include <random>
#include <utility>
#include <vector>

#include "tree_builder.h"

using namespace std;
// -----------------------------------------------------------------------------



/*
    This file defines the TreeOperators class, which implements the genetic
    operators for manipulating expression trees in the GP algorithm:

      - crossover : swaps two random subtrees between copies of the parents.
      - mutate    : point mutation (terminal reindex / same-arity operator
                    swap) plus occasional subtree replacement.
      - hoist     : replaces the whole tree with one of its own subtrees.

    Offspring invariants, enforced by every operator that can grow a tree:
      - node count  ≤ maxTreeSize()  ( = 2^(maxDepth+1) )
      - tree depth  ≤ maxTreeDepth_
    A child that violates either cap is discarded and replaced by a copy of
    its parent (standard GP practice), NOT by a fresh random tree — the old
    behaviour threw away both parents' genetic material.

    LITERATURE NOTES  (keys refer to thesis/references.bib; section numbers are
    those of the cited papers, see thesis/literature_notes.md)
    ----------------------------------------------------------------------
    * Subtree crossover.  Two parents each give a randomly chosen subtree and
      the offspring are produced by swapping them (Koza 1994, koza1994gp, §3;
      Sun et al. 2024, mtgp_multicloud, §4.1 and Fig. 3a; Zhang & Yang 2025,
      zhang2025mtgper, §3.2.1).  This project has a single tree per individual,
      so the multi-tree variants of crossover (swap a whole second tree, as in
      the sMTGP of Zhang et al. 2018, zhang2018mtgp, and in zhang2025mtgper
      §3.2.1) have no counterpart here.

    * Crossover point bias.  Koza evolves with 90% of the crossover points
      internal (function) nodes and 10% external (terminal) nodes (koza1994gp,
      §4, secondary parameters).  In a tree with binary operators about half of
      the nodes are leaves, so picking a node uniformly makes roughly one
      crossover in four swap two bare terminals, which is only a point
      mutation.  The bias is available as `setInternalCrossoverBias()`; it is
      0 (uniform, the pre-existing behaviour) unless SchedulerGP turns it on.
      The scheduling papers (mtgp_multicloud, zhang2025mtgper, gp_fog) say only
      that a node is selected at random and do not state a bias.

    * Subtree mutation.  "Generate a new subtree, then replace a random subtree
      of the parent with it" is the mutation of mtgp_multicloud §4.1 (Fig. 3b),
      zhang2025mtgper §3.2.1 and gp_fog §VI-A.5.  The 10% branch of
      mutateNode() below is that operator, with a depth-3 grow tree.  The
      per-node point mutation (terminal redraw, same-arity operator swap) is an
      addition of this project and is not taken from those papers.

    * Operator probabilities.  In the three papers crossover, mutation and
      reproduction are *alternatives* chosen per offspring (0.8 / 0.15 / 0.05 in
      mtgp_multicloud Table 6, zhang2025mtgper Table 3 and the DTGP baseline
      of Yang et al. 2025, yang2025thesis, Table 3.7).  Here crossover (0.8) and mutation
      are applied one after the other, so an offspring can receive both;
      mutationRate is per node, not per individual.

    * Depth cap.  The papers bound tree depth (8 in mtgp_multicloud Table 6 and
      dtgpam §V-C, 7 in zhang2025mtgper Table 3; Koza uses 17 for crossover
      offspring, koza1994gp §4) and do nothing else against bloat.  The extra
      node-count cap and the hoist mutation below are this project's additions.
*/



class TreeOperators {
public:
    #pragma region CONSTRUCTORS ------------------------------------------------
    explicit TreeOperators(mt19937& rng, TreeBuilder& builder, int maxDepth = 7)
        : rng_(rng), builder_(builder), maxTreeDepth_(maxDepth) {}
    #pragma endregion ----------------------------------------------------------



    #pragma region PUBLIC API --------------------------------------------------
    // ── Node location ────────────────────────────────────────────────────────
    // Preorder-index navigation.  Returns a pointer instead of a value so the
    // located subtree is copied at most once, at the call site (the previous
    // by-value helpers re-copied the found subtree at every recursion level
    // on the way back up).

    static ExprNode* locate(ExprNode& root, int target) {
        int cur = 0;
        return locateAt(root, target, cur);
    }
    static const ExprNode* locate(const ExprNode& root, int target) {
        return locate(const_cast<ExprNode&>(root), target);
    }

    // ── Subtree crossover ────────────────────────────────────────────────────
    // Copy both parents once, then *swap* the two randomly chosen subtrees
    // between the copies.  This produces exactly the same two children as the
    // old extract-and-substitute implementation, with zero extra subtree
    // copies.
    pair<ExprNode, ExprNode> crossover(const ExprNode& a, const ExprNode& b) {
        ExprNode childA = a;
        ExprNode childB = b;

        // Crossover points: uniform by default, or biased towards internal
        // nodes (Koza 1994 §4: 90% internal / 10% terminal).  Parent A's
        // point is drawn before parent B's, as before.
        ExprNode* nodeA = locate(childA, pickCrossoverPoint(childA));
        ExprNode* nodeB = locate(childB, pickCrossoverPoint(childB));
        swap(*nodeA, *nodeB);

        // Enforce the size/depth caps by reverting to the parent.  This is the
        // role of Koza's depth limit on crossover offspring (koza1994gp §4:
        // initial programs depth 6, crossover offspring depth 17).  NOTE: if
        // the initial trees are generated at the cap itself, any crossover that
        // grows a tree is reverted, which is why the papers start below the
        // cap (see TreeBuilder::rampedHalfAndHalf).
        if (violatesLimits(childA)) childA = a;
        if (violatesLimits(childB)) childB = b;

        // No simplify() here: in the breeding loop mutate() always runs right
        // after crossover and finishes with a single simplify pass, so
        // simplifying twice per offspring was pure duplicated work.
        return {childA, childB};
    }

    // ── Point mutation ───────────────────────────────────────────────────────
    // Recursively walk and return a mutated copy.  This avoids UAF from raw
    // pointer lists becoming stale after a child vector reallocation.
    //
    // When mutation fires on a node:
    //   TERMINAL  → re-draw the terminal index.
    //   CONST     → demote to a terminal (constants are disabled).
    //   operator  → 10%: replace the whole subtree with a fresh grow(3) tree;
    //               90%: swap the operator for a *different* operator of the
    //                    same arity (children preserved), then keep walking.
    //               (Previously the 90% branch did nothing at all, so an ADD
    //               could never become a SUB and the effective mutation rate
    //               on internal nodes was mutationRate × 0.10.)
    ExprNode mutateNode(ExprNode node, double mutationRate) {
        if (randomReal() < mutationRate) {
            switch (node.nodeType) {
                case NodeType::TERMINAL:
                    // Same draw as a freshly grown leaf: uniform unless the
                    // builder carries an active-terminal set or a learned
                    // distribution (SchedulerGP terminal weighting).
                    node.terminalIndex = builder_.randomTerminalIndex();
                    return node;
                case NodeType::CONST:
                    // Demote any stale CONST node (e.g. loaded from an external
                    // source) to a proper terminal so the invariant is enforced.
                    node.nodeType   = NodeType::TERMINAL;
                    node.constValue = 0.0;
                    node.terminalIndex = builder_.randomTerminalIndex();
                    node.children.clear();
                    return node;
                default:
                    if (randomReal() < 0.10)
                        return builder_.grow(3);
                    node.nodeType = randomOperatorSameArity(node.nodeType);
                    break;
            }
        }
        for (auto& child : node.children)
            child = mutateNode(move(child), mutationRate);
        return node;
    }

    void mutate(ExprNode& tree, double mutationRate) {
        ExprNode original = tree;   // revert target if the caps are violated
        tree = mutateNode(move(tree), mutationRate);
        tree.simplifyInPlace();     // may shrink an oversized tree back under the caps
        if (violatesLimits(tree)) tree = move(original);
    }

    // ── Hoist mutation ───────────────────────────────────────────────────────
    // Replace the whole tree with one of its own non-root subtrees, chosen
    // uniformly.  Locates the subtree by preorder index and moves it out —
    // the previous version materialised a copy of *every* subtree into a
    // vector (O(n·depth) node copies) just to pick one.
    void hoist(ExprNode& tree) {
        const int n = tree.size();
        if (n <= 1) return;
        ExprNode* sub = locate(tree, randomInt(1, n - 1));
        ExprNode promoted = move(*sub);
        tree = move(promoted);
    }

    /// @brief Probability that a crossover point is an internal (operator)
    /// node.  0 (default) keeps the uniform choice over all nodes; Koza uses
    /// 0.9 (koza1994gp §4).  Values are clamped to [0, 1].
    void setInternalCrossoverBias(double p) {
        internalCrossoverBias_ = std::max(0.0, std::min(1.0, p));
    }

    void setTerminalCount(int n) { terminalCount_ = n; }
    void setMaxDepth(int d)      { maxTreeDepth_  = std::max(1, d); }
    #pragma endregion ----------------------------------------------------------



private:
    #pragma region FIELDS ------------------------------------------------------
    mt19937&     rng_;
    TreeBuilder& builder_;
    int          maxTreeDepth_;
    int          terminalCount_ = 1;
    double       internalCrossoverBias_ = 0.0;
    #pragma endregion ----------------------------------------------------------



    #pragma region UTILITIES ---------------------------------------------------
    static ExprNode* locateAt(ExprNode& node, int target, int& cur) {
        if (cur == target) return &node;
        for (auto& c : node.children) {
            ++cur;
            if (ExprNode* r = locateAt(c, target, cur)) return r;
        }
        return nullptr;
    }

    /// @brief Preorder indices of the operator (internal) and terminal nodes.
    static void collectIndices(const ExprNode& n, int& cur,
                               vector<int>& internal, vector<int>& leaves) {
        (n.children.empty() ? leaves : internal).push_back(cur);
        for (const auto& c : n.children) {
            ++cur;
            collectIndices(c, cur, internal, leaves);
        }
    }

    /// @brief Chooses the preorder index of a crossover point.  With no bias
    /// this is exactly the original single uniform draw, so seeded runs are
    /// unchanged.
    int pickCrossoverPoint(const ExprNode& tree) {
        if (internalCrossoverBias_ <= 0.0) return randomInt(0, tree.size() - 1);

        vector<int> internal, leaves;
        int cur = 0;
        collectIndices(tree, cur, internal, leaves);
        if (internal.empty()) return 0;            // a lone terminal: only the root
        const bool pickInternal = randomReal() < internalCrossoverBias_ || leaves.empty();
        const vector<int>& pool = pickInternal ? internal : leaves;
        return pool[randomInt(0, static_cast<int>(pool.size()) - 1)];
    }

    int maxTreeSize() const { return 1 << (maxTreeDepth_ + 1); }

    bool violatesLimits(const ExprNode& t) const {
        return t.size() > maxTreeSize() || t.depth() > maxTreeDepth_;
    }

    /// @brief Uniformly picks an operator of the same arity as `t`, excluding
    /// `t` itself (single RNG draw, so the stream stays deterministic).
    NodeType randomOperatorSameArity(NodeType t) {
        const bool      binary = ARITY[static_cast<int>(t)] == 2;
        const NodeType* ops    = binary ? TreeBuilder::kBinaryOps.data()
                                        : TreeBuilder::kUnaryOps.data();
        const int       n      = binary ? static_cast<int>(TreeBuilder::kBinaryOps.size())
                                        : static_cast<int>(TreeBuilder::kUnaryOps.size());
        int idx = randomInt(0, n - 2);
        if (ops[idx] == t) idx = n - 1;
        return ops[idx];
    }

    double randomReal() {
        uniform_real_distribution<double> d{0.0, 1.0};
        return d(rng_);
    }
    int randomInt(int lo, int hi) {
        uniform_int_distribution<int> d{lo, hi};
        return d(rng_);
    }
    #pragma endregion ----------------------------------------------------------
};
