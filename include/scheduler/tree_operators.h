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

        ExprNode* nodeA = locate(childA, randomInt(0, childA.size() - 1));
        ExprNode* nodeB = locate(childB, randomInt(0, childB.size() - 1));
        swap(*nodeA, *nodeB);

        // Enforce the size/depth caps by reverting to the parent.
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
                    node.terminalIndex = randomInt(0, terminalCount_ - 1);
                    return node;
                case NodeType::CONST:
                    // Demote any stale CONST node (e.g. loaded from an external
                    // source) to a proper terminal so the invariant is enforced.
                    node.nodeType   = NodeType::TERMINAL;
                    node.constValue = 0.0;
                    node.terminalIndex = randomInt(0, terminalCount_ - 1);
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

    void setTerminalCount(int n) { terminalCount_ = n; }
    void setMaxDepth(int d)      { maxTreeDepth_  = std::max(1, d); }
    #pragma endregion ----------------------------------------------------------



private:
    #pragma region FIELDS ------------------------------------------------------
    mt19937&     rng_;
    TreeBuilder& builder_;
    int          maxTreeDepth_;
    int          terminalCount_ = 1;
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
