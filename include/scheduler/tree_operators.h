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

    * What happens to an offspring that exceeds the cap.  By default it is
      discarded and a copy of its parent is returned, which is Koza's
      implementation of the limit.  Poli, Langdon & McPhee (poli2008fieldguide,
      §11.3.2 "Size and Depth Limits", pp. 104-105) call this "a serious
      problem": parents that are likely to violate the limit get copied
      unaltered more often, so the population fills with programs that nearly
      infringe it (bushy, almost full trees for a depth limit).  Their remedy
      is not to return the parent: either keep the offspring with fitness 0,
      or declare the operation failed and try again, (a) with the same parents
      and new crossover or mutation points, or (b) with new parents.
      Two opt-in options implement remedy (a) in closed form, i.e. without a
      retry loop and without ever producing an invalid tree:

        - depth-aware crossover (setDepthAwareCrossover): the first point is
          drawn as usual, the second among the nodes for which both children
          respect the cap.  That is the distribution "redraw the second point
          until the children fit" would give.  Constraining the second point
          by what the first one removed is also how size-fair crossover works
          (Langdon 2000, langdon2000sizefair; poli2008fieldguide §11.3.2,
          "Anti-bloat genetic operators"), with a size criterion there and a
          depth criterion here.  A fitting node always exists (Proposition 3,
          thesis/chapters/method.tex).
        - depth-aware mutation (setDepthAwareMutation): the replacement
          subtree is grown within the depth left below the cap.  The same
          section cites mutation operators that bound the offspring's depth
          (Kinnear 1993) or match the size of the removed code (Langdon 1998);
          the exact form used here, grow(min(3, D - depth)), is this project's.
          Without it one oversized replacement throws away every point
          mutation made in the same tree (mutate() reverts the whole tree).

      NOT YET MEASURED on the held-out worlds (run_ablation.py, configs
      "depthmut", "depthx", "depthops"); both default to off, so seeded runs
      and the thesis results are unchanged.
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
        const int indexA = pickCrossoverPoint(childA);
        // Depth-aware crossover (opt-in): parent B's point is drawn only
        // among the nodes for which BOTH children respect the depth cap, so
        // the revert below never fires.  -1 = no such node (cannot happen for
        // parents within the cap, see pickFittingPoint) or option off.
        int indexB = depthAwareCrossover_ ? pickFittingPoint(childA, indexA, childB) : -1;
        if (indexB < 0) indexB = pickCrossoverPoint(childB);
        ExprNode* nodeA = locate(childA, indexA);
        ExprNode* nodeB = locate(childB, indexB);
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
    //
    // `depth` is the depth of `node` in the whole tree (root = 0).  It is used
    // only by depth-aware mutation (opt-in): the replacement subtree is then
    // grown within the room that is left below the cap,
    //     grow( min(3, maxTreeDepth - depth) ),
    // so the mutated tree can never exceed the cap and mutate() never has to
    // throw the whole mutation away.  Off (default) the subtree is grow(3)
    // wherever the node is, exactly as before.
    ExprNode mutateNode(ExprNode node, double mutationRate, int depth = 0) {
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
                    if (randomReal() < 0.10) {
                        const int room = depthAwareMutation_
                            ? std::max(0, std::min(3, maxTreeDepth_ - depth))
                            : 3;
                        return builder_.grow(room);
                    }
                    node.nodeType = randomOperatorSameArity(node.nodeType);
                    break;
            }
        }
        for (auto& child : node.children)
            child = mutateNode(move(child), mutationRate, depth + 1);
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

    /// @brief Depth-aware subtree mutation (default off).  The replacement
    /// subtree is grown within the depth left below the cap instead of always
    /// at depth 3.  See mutateNode() and the LITERATURE NOTES at the top.
    void setDepthAwareMutation(bool on)  { depthAwareMutation_  = on; }

    /// @brief Depth-aware crossover (default off).  The second crossover
    /// point is drawn among the nodes for which both children respect the
    /// depth cap.  See pickFittingPoint() and the LITERATURE NOTES at the top.
    void setDepthAwareCrossover(bool on) { depthAwareCrossover_ = on; }

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
    bool         depthAwareMutation_  = false;
    bool         depthAwareCrossover_ = false;
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

    /// @brief Depth (distance from the root) and height (depth of its own
    /// subtree) of every node, in preorder.
    struct NodeInfo {
        int  depth;
        int  height;
        bool internal;
    };

    /// @brief Fills `out` in preorder; returns the height of `n`.
    static int annotate(const ExprNode& n, int depth, vector<NodeInfo>& out) {
        const size_t self = out.size();
        out.push_back({depth, 0, !n.children.empty()});
        int height = 0;
        for (const auto& c : n.children)
            height = std::max(height, 1 + annotate(c, depth + 1, out));
        out[self].height = height;
        return height;
    }

    /// @brief Depth-aware choice of parent B's crossover point, given parent
    /// A's point `indexA`.  Returns a preorder index in `b`, or -1 if no node
    /// fits (then the caller falls back to the ordinary draw).
    ///
    /// Swapping the subtree at u (in a) with the subtree at v (in b) gives
    ///     depth(child A) = max( depth of a without u's subtree, d_a(u) + h_b(v) )
    ///     depth(child B) = max( depth of b without v's subtree, d_b(v) + h_a(u) )
    /// (d = depth of the node, h = height of its subtree).  The parents are
    /// within the cap D, so the children are within it exactly when
    ///     d_a(u) + h_b(v) <= D     and     d_b(v) + h_a(u) <= D.
    /// v is drawn among the nodes of b that satisfy both.  Such a node always
    /// exists when both parents respect the cap (Proposition 3,
    /// thesis/chapters/method.tex): on the path from b's root to one of its
    /// deepest leaves, take the node at depth max(0, depth(b) - D + d_a(u)).
    ///
    /// Distribution.  This is the ordinary draw of v conditioned on "both
    /// children fit", i.e. what "draw v again until the children fit" would
    /// return, without the loop: uniform over the fitting nodes when there is
    /// no internal-node bias; with a bias p, the internal and the leaf pools
    /// keep their weights p and 1 - p, each scaled by the fraction of its
    /// nodes that fit.  [poli2008fieldguide §11.3.2, strategy "same parents,
    /// new crossover points"; langdon2000sizefair for constraining the second
    /// point by what the first one removed.]
    int pickFittingPoint(const ExprNode& a, int indexA, const ExprNode& b) {
        vector<NodeInfo> infoA, infoB;
        annotate(a, 0, infoA);
        annotate(b, 0, infoB);
        if (indexA < 0 || indexA >= static_cast<int>(infoA.size())) return -1;
        const int depthA  = infoA[indexA].depth;
        const int heightA = infoA[indexA].height;

        vector<int> fitInternal, fitLeaves;
        int nInternal = 0, nLeaves = 0;
        for (int v = 0; v < static_cast<int>(infoB.size()); ++v) {
            const NodeInfo& nb = infoB[v];
            (nb.internal ? nInternal : nLeaves) += 1;
            if (depthA + nb.height <= maxTreeDepth_ && nb.depth + heightA <= maxTreeDepth_)
                (nb.internal ? fitInternal : fitLeaves).push_back(v);
        }
        if (fitInternal.empty() && fitLeaves.empty()) return -1;

        // Weight of each pool under the ordinary draw, restricted to fitting nodes.
        double wInternal, wLeaves;
        if (internalCrossoverBias_ <= 0.0) {            // uniform over all nodes
            wInternal = static_cast<double>(fitInternal.size());
            wLeaves   = static_cast<double>(fitLeaves.size());
        } else {
            // pickCrossoverPoint takes the internal pool with probability p,
            // the leaf pool otherwise; a lone terminal has no internal pool.
            const double p = (nInternal == 0) ? 0.0 : internalCrossoverBias_;
            wInternal = (nInternal > 0) ? p * fitInternal.size() / nInternal : 0.0;
            wLeaves   = (nLeaves   > 0) ? (1.0 - p) * fitLeaves.size() / nLeaves : 0.0;
            if (wInternal + wLeaves <= 0.0) {           // e.g. p = 1 and only leaves fit
                wInternal = static_cast<double>(fitInternal.size());
                wLeaves   = static_cast<double>(fitLeaves.size());
            }
        }
        const bool takeInternal =
            !fitInternal.empty() &&
            (fitLeaves.empty() || randomReal() * (wInternal + wLeaves) < wInternal);
        const vector<int>& pool = takeInternal ? fitInternal : fitLeaves;
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
