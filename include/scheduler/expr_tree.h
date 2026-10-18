#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"
#include "registry.h"
// -----------------------------------------------------------------------------



/*
    This file defines the class ExprNode, which represents a node in the 
    expression tree used for scoring jobs in the genetic programming algorithm.

    Each job-satellite pair is scored by evaluating the expression tree, where 
    the terminals are features of the job and satellite (e.g., job priority, 
    satellite energy). These scoring are used as scheduling criteria. The GP 
    algorithm evolves these trees to find effective trees that lead to good
    scheduling criteria.

    Examples of expressions that might be evolved include:
        - `priority / (1 + activeTasks)` to prefer high-priority jobs on less 
          busy satellites.
        - `priority * (1 + RemainingEnergy)` to prefer high-priority jobs on 
          satellites with more energy.
        - `priority / (1 + abs(dueMinute - currentTime))` to prefer jobs that 
          are closer to their due time.
    In these cases, "priority", "activeTasks", "RemainingEnergy", and 
    "dueMinute" would be terminals in the expression tree, and the tree would 
    combine them using operators like +, -, *, /, etc.
*/



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
            {
                double result = registry[terminalIndex].accessor(job, sat);
                return result;
            }
            case NodeType::CONST:
            {
                double result = constValue;
                return result;
            }
            case NodeType::ADD:
            {
                double result = children[0].eval(job, sat, registry) + children[1].eval(job, sat, registry);
                return result;
            }
            case NodeType::SUB:
            {
                double result = children[0].eval(job, sat, registry) - children[1].eval(job, sat, registry);
                return result;
            }
            case NodeType::MUL:
            {
                double result = children[0].eval(job, sat, registry) * children[1].eval(job, sat, registry);
                return result;
            }
            case NodeType::DIV: {
                double denom = children[1].eval(job, sat, registry);
                double result = (abs(denom) < 1e-9) ? 0.0 : children[0].eval(job, sat, registry) / denom;
                return result;
            }
            case NodeType::MAX_OP:
            {
                double result = max(children[0].eval(job, sat, registry), children[1].eval(job, sat, registry));
                return result;
            }
            case NodeType::MIN_OP:
            {
                double result = min(children[0].eval(job, sat, registry), children[1].eval(job, sat, registry));
                return result;
            }
            case NodeType::NEG:
            {
                double result = -children[0].eval(job, sat, registry);
                return result;
            }
            case NodeType::ABS:
            {
                double result = abs(children[0].eval(job, sat, registry));
                return result;
            }
            case NodeType::INV: {
                double v = children[0].eval(job, sat, registry);
                double result = (abs(v) < 1e-9) ? 0.0 : 1.0 / v;
                return result;
            }
        }

        throw runtime_error("Invalid node type" + to_string(static_cast<int>(nodeType)) + " in ExprNode::eval");
    }
    #pragma endregion ----------------------------------------------------------



    #pragma region UTILITIES ---------------------------------------------------
    /// @brief Returns true iff the two subtrees are structurally identical
    /// (same topology, same node types, same terminal indices / const values).
    static bool structurallyEqual(const ExprNode& a, const ExprNode& b) {
        if (a.nodeType != b.nodeType)              return false;
        if (a.nodeType == NodeType::TERMINAL)      return a.terminalIndex == b.terminalIndex;
        if (a.nodeType == NodeType::CONST)         return a.constValue    == b.constValue;
        if (a.children.size() != b.children.size()) return false;
        for (size_t i = 0; i < a.children.size(); ++i)
            if (!structurallyEqual(a.children[i], b.children[i])) return false;
        return true;
    }

    /// @brief Simplifies this subtree **in place**, collapsing provably
    /// redundant operation patterns bottom-up.
    ///
    /// Every rule is *exact* under eval()'s guarded semantics — simplifying a
    /// tree can never change the score it produces for any (job, satellite)
    /// pair.  Two former rules were removed for exactly that reason:
    ///   DIV(x, x)    → CONST(1)  is wrong when x ≈ 0 (guarded div yields 0),
    ///   INV(INV(x))  → x         is wrong when |x| ≥ ~1e9 or |x| ≤ ~1e-9
    ///                              (the inner guarded INV collapses to 0).
    ///
    /// Rules applied (after children are already simplified):
    ///   NEG(NEG(x))  → x
    ///   ABS(ABS(x))  → ABS(x)
    ///   ABS(NEG(x))  → ABS(x)     (both ABS rules iterated to a fixpoint)
    ///   MAX(x, x)    → x
    ///   MIN(x, x)    → x
    ///   SUB(x, x)    → CONST(0)
    ///
    /// In-place on purpose: the previous copy-based simplify() deep-copied the
    /// whole subtree at every recursion level (O(n·depth) node copies per
    /// call), and it ran for every offspring every generation.
    void simplifyInPlace() {
        // --- 1. Recurse: simplify children first.
        for (auto& child : children)
            child.simplifyInPlace();

        // --- 2. Apply local rules on the (now-simplified) node.
        switch (nodeType) {

            // ── NEG ──────────────────────────────────────────────────────────
            case NodeType::NEG: {
                // NEG(NEG(x)) → x
                if (children[0].nodeType == NodeType::NEG) {
                    ExprNode hoisted = move(children[0].children[0]);
                    *this = move(hoisted);
                }
                break;
            }

            // ── ABS ──────────────────────────────────────────────────────────
            case NodeType::ABS: {
                // ABS(ABS(x)) → ABS(x),  ABS(NEG(x)) → ABS(x).
                // Stripping a NEG can expose a new ABS/NEG directly underneath,
                // so iterate until neither rule matches.
                while (children[0].nodeType == NodeType::ABS ||
                       children[0].nodeType == NodeType::NEG) {
                    ExprNode hoisted = move(children[0].children[0]);
                    children[0] = move(hoisted);
                }
                break;
            }

            // ── MAX / MIN ────────────────────────────────────────────────────
            case NodeType::MAX_OP:
            case NodeType::MIN_OP: {
                // MAX(x, x) → x,  MIN(x, x) → x
                if (structurallyEqual(children[0], children[1])) {
                    ExprNode hoisted = move(children[0]);
                    *this = move(hoisted);
                }
                break;
            }

            // ── SUB ──────────────────────────────────────────────────────────
            case NodeType::SUB: {
                // SUB(x, x) → CONST(0)
                if (structurallyEqual(children[0], children[1])) {
                    nodeType      = NodeType::CONST;
                    constValue    = 0.0;
                    terminalIndex = -1;
                    children.clear();
                }
                break;
            }

            default:
                break;
        }
    }

    /// @brief Returns a simplified copy of this subtree (see simplifyInPlace).
    ExprNode simplify() const {
        ExprNode node = *this;
        node.simplifyInPlace();
        return node;
    }

    /// @brief Count the number of nodes in the subtree.
    /// @return The number of nodes.
    int size() const {
        int s = 1;
        for (const auto& c : children) s += c.size();
        return s;
    }

    /// @brief Depth of the subtree in edges: a lone terminal has depth 0.
    /// Matches TreeBuilder's convention (grow/full(maxDepth) produce trees
    /// whose depth() is at most maxDepth).
    int depth() const {
        int d = 0;
        for (const auto& c : children) d = max(d, 1 + c.depth());
        return d;
    }

    /// @brief Returns the expression represented by this node as a string.
    /// @param registry The terminal registry to resolve terminal names.
    /// @return The string representation of the expression.
    /// @throws runtime_error if the node type is invalid.
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

        throw runtime_error("Invalid node type" + to_string(static_cast<int>(nodeType)) + " in ExprNode::toString");
    }
    #pragma endregion ----------------------------------------------------------
};