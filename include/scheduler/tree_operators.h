#pragma once

#include <random>
#include <utility>
#include <vector>

#include "tree_builder.h"

using namespace std;
// -----------------------------------------------------------------------------



/*
    This file defines the TreeOperators class, which implements the genetic
    operators for manipulating expression trees in the GP algorithm.
*/



class TreeOperators {
public:
    #pragma region CONSTRUCTORS ------------------------------------------------
    explicit TreeOperators(mt19937& rng, TreeBuilder& builder, int maxDepth = 7)
        : rng_(rng), builder_(builder), maxTreeDepth_(maxDepth) {}
    #pragma endregion ----------------------------------------------------------



    #pragma region PUBLIC API --------------------------------------------------
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
    #pragma endregion ----------------------------------------------------------

    
    
private:
    #pragma region FIELDS ------------------------------------------------------
    mt19937&     rng_;
    TreeBuilder& builder_;
    int          maxTreeDepth_;
    int          terminalCount_ = 1;
    #pragma endregion ----------------------------------------------------------



    #pragma region UTILITIES ---------------------------------------------------
    int maxTreeSize() const { return 1 << (maxTreeDepth_ + 1); }

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
