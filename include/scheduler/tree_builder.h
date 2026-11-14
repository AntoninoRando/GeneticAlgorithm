#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>
#include "expr_tree.h"

using namespace std;
// -----------------------------------------------------------------------------


/*
    This file defines the TreeBuilder class, which is responsible for generating
    random expression trees for the genetic programming algorithm. It provides
    three methods for building trees:
        1. grow: at each internal node, randomly pick operator or terminal.
        2. full: every leaf is at exactly maxDepth.
        3. rampedHalfAndHalf: builds a population of  'count' trees with a mix
           of grow and full, and varying depths.

    LITERATURE NOTES  (keys refer to thesis/references.bib; section numbers are
    those of the cited papers, see thesis/literature_notes.md)
    ----------------------------------------------------------------------
    * Why ramped half-and-half?  A random initial population should contain
      "a wide variety of different sizes and shapes" (Koza 1994, koza1994gp,
      §4, secondary parameters).  Every GP-for-scheduling paper reviewed here
      initialises with it: Sun et al. 2024 (mtgp_multicloud) §4.1, Zhang & Yang
      2025 (zhang2025mtgper) §3.2.1, Xu et al. 2023 (gp_fog) §VI-A.2.  Half the
      trees come from `full` and half from `grow`, with the depth ramped across
      the population.

    * Function set.  The six binary operators {+, -, *, /, max, min} are exactly
      the function set of mtgp_multicloud §4.2 and zhang2025mtgper §3.2.2.  The
      three unary operators {neg, abs, inv} are additions of this project.

    * Closure.  GP needs every function to be defined for every value it can
      receive (koza1994gp §4, "closure").  The papers obtain it with a
      *protected division that returns 1* when the denominator is 0
      (mtgp_multicloud §4.2, zhang2025mtgper §3.2.2).  This project's
      ExprNode::eval returns 0 instead (and does the same for inv); closure
      holds either way.  The difference only matters for how a degenerate
      denominator ranks a pair: 0 puts it in the middle of a score range that
      is mostly positive, 1 would not.  Kept at 0 so that already evolved
      trees and the thesis results keep their meaning.
*/



class TreeBuilder {
public:
    #pragma region CONSTRUCTORS ------------------------------------------------
    explicit TreeBuilder(const vector<TerminalDef>& registry, mt19937& rng)
        : registry_(registry), rng_(rng) {}

    /// Operator sets, shared by every builder (also used by TreeOperators for
    /// same-arity point mutation).  Static so that constructing a TreeBuilder
    /// is allocation-free — one is now created per offspring pair in the
    /// parallel breeding loop.
    static constexpr array<NodeType, 6> kBinaryOps = {
        NodeType::ADD, NodeType::SUB, NodeType::MUL,
        NodeType::DIV, NodeType::MAX_OP, NodeType::MIN_OP };
    static constexpr array<NodeType, 3> kUnaryOps = {
        NodeType::NEG, NodeType::ABS, NodeType::INV };
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
    //
    // NOTE: this is a *biased* grow, not the textbook one.  With 3 or more
    // levels left the node is always an operator; only below that is a
    // terminal chosen early (55%).  The bias keeps `grow` trees from
    // collapsing into one or two nodes when called with a small depth, e.g.
    // for the depth-3 replacement subtrees of subtree mutation.
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
    //
    // `minDepth`..`maxDepth` is the range of the *initial* trees.  The papers
    // keep the initial maximum depth below the evolution cap so crossover has
    // headroom to grow a tree: Sun et al. 2024 start at depth 2-7 with a cap
    // of 8 (Table 6), Yang et al. 2025 at 2-6 with a cap of 8 (dtgpam §V-C), Zhang & Yang 2025 at 1-3 with a cap of 7 (Table 3), and Koza
    // 1994 at 6 with a cap of 17 (§4).  SchedulerGP::initialize() exposes this
    // as `initialMaxDepth`.
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

    /// @brief Draws one terminal index.  This is the single place where a
    /// terminal is chosen, for initial trees, replacement subtrees and the
    /// point mutation of a terminal leaf (TreeOperators::mutateNode).
    ///
    /// By default the draw is uniform over the whole registry, consuming the
    /// RNG exactly as the original code did, so seeded runs are unchanged.
    /// Two optional restrictions can be installed, both used only when the
    /// caller asks for them:
    ///   - an *active set* (setActiveTerminals): terminals that are constant
    ///     in the data are never drawn;
    ///   - a *distribution* (setTerminalDistribution): a learned,
    ///     non-uniform terminal probability (Yang et al. 2025 terminal
    ///     weighting, see SchedulerGP).
    int randomTerminalIndex() {
        if (terminalCdf_ != nullptr && !terminalCdf_->empty()) {
            const double u  = randomReal();
            const auto   it = lower_bound(terminalCdf_->begin(), terminalCdf_->end(), u);
            const size_t k  = min(static_cast<size_t>(it - terminalCdf_->begin()),
                                  terminalCdf_->size() - 1);
            return (activeTerminals_ != nullptr && !activeTerminals_->empty())
                       ? (*activeTerminals_)[k]
                       : static_cast<int>(k);
        }
        if (activeTerminals_ != nullptr && !activeTerminals_->empty()) {
            return (*activeTerminals_)[randomInt(0, static_cast<int>(activeTerminals_->size()) - 1)];
        }
        return randomInt(0, static_cast<int>(registry_.size()) - 1);
    }

    /// @brief Restricts every terminal draw to `active` (registry indices).
    /// Pointer is non-owning and must outlive the builder; nullptr or an
    /// empty vector restores the full registry.
    void setActiveTerminals(const vector<int>* active) { activeTerminals_ = active; }

    /// @brief Installs a cumulative distribution over the active terminals
    /// (same order as the active set, or the full registry when no active set
    /// is installed).  Last entry must be 1.  nullptr restores uniform draws.
    void setTerminalDistribution(const vector<double>* cdf) { terminalCdf_ = cdf; }
    #pragma endregion ----------------------------------------------------------



private:
    #pragma region FIELDS ------------------------------------------------------
    const vector<TerminalDef>& registry_;
    mt19937& rng_;
    const vector<int>*    activeTerminals_ = nullptr;
    const vector<double>* terminalCdf_     = nullptr;
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
        // Constants are disabled: every leaf is a registry terminal.  The papers
        // reviewed also use terminals only (no ephemeral constants):
        // mtgp_multicloud §4.2 (Table 2), zhang2025mtgper §3.2.2 (Table 2).
        node.nodeType      = NodeType::TERMINAL;
        node.terminalIndex = randomTerminalIndex();
        return node;
    }

    ExprNode makeOperatorNode(int maxDepth, bool forceOperator) {
        ExprNode node;

        // Choose arity: binary ~70%, unary ~30%.
        bool binary = (randomReal() < 0.70);
        if (binary) {
            node.nodeType = kBinaryOps[randomInt(0, static_cast<int>(kBinaryOps.size()) - 1)];
            if (forceOperator) {
                node.children.push_back(full(maxDepth - 1));
                node.children.push_back(full(maxDepth - 1));
            } else {
                node.children.push_back(grow(maxDepth - 1));
                node.children.push_back(grow(maxDepth - 1));
            }
        } else {
            node.nodeType = kUnaryOps[randomInt(0, static_cast<int>(kUnaryOps.size()) - 1)];
            if (forceOperator) node.children.push_back(full(maxDepth - 1));
            else               node.children.push_back(grow(maxDepth - 1));
        }
        return node;
    }
    #pragma endregion ----------------------------------------------------------
};
