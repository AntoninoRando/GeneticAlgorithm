#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include "expr_tree.h"

using namespace std;
// -----------------------------------------------------------------------------



class TreeBuilder {
public:
    #pragma region CONSTRUCTORS ------------------------------------------------
    explicit TreeBuilder(const vector<TerminalDef>& registry, mt19937& rng)
        : registry_(registry), rng_(rng)
    {
        // Binary operators
        binaryOps_ = { NodeType::ADD, NodeType::SUB, NodeType::MUL,
                       NodeType::DIV, NodeType::MAX_OP, NodeType::MIN_OP };
        // Unary operators
        unaryOps_  = { NodeType::NEG, NodeType::ABS, NodeType::INV };
    }
    #pragma endregion ----------------------------------------------------------



    #pragma region PUBLIC API --------------------------------------------------
    /*
        This section provides three methods for building expression trees:
        1. grow: at each internal node, randomly pick operator or terminal.
        2. full: every leaf is at exactly maxDepth.
        3. rampedHalfAndHalf: builds a population of  'count' trees with a mix 
           of grow and full, and varying depths.
    */

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
    #pragma endregion ----------------------------------------------------------



private:
    #pragma region FIELDS ------------------------------------------------------
    const vector<TerminalDef>& registry_;
    mt19937& rng_;
    vector<NodeType> binaryOps_;
    vector<NodeType> unaryOps_;
    #pragma endregion ----------------------------------------------------------



    #pragma region UTILITIES ---------------------------------------------------
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
    #pragma endregion ----------------------------------------------------------
};
