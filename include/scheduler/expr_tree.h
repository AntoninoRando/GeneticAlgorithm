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

    /// @brief Returns a semantically equivalent, simplified copy of this
    /// subtree, collapsing provably redundant operation patterns bottom-up.
    ///
    /// Rules applied (in order, after children are already simplified):
    ///
    /// Unary identity chains
    ///   NEG(NEG(x))  → x
    ///   ABS(ABS(x))  → ABS(x)
    ///   ABS(NEG(x))  → ABS(x)
    ///   INV(INV(x))  → x
    ///
    /// Same-child binary idempotency
    ///   MAX(x, x)    → x
    ///   MIN(x, x)    → x
    ///   SUB(x, x)    → CONST(0)
    ///   DIV(x, x)    → CONST(1)   (inherits guarded-div: if x≈0 eval→0, not 1;
    ///                               but that pre-existing corner case is unchanged)
    ///
    /// @return The simplified node (may be a different node type / shape).
    ExprNode simplify() const {
        // --- 1. Recurse: build a copy with simplified children first.
        ExprNode node = *this;
        for (auto& child : node.children)
            child = child.simplify();

        // --- 2. Apply local rules on the (now-simplified) node.
        switch (node.nodeType) {

            // ── NEG ──────────────────────────────────────────────────────────
            case NodeType::NEG: {
                const ExprNode& inner = node.children[0];
                // NEG(NEG(x)) → x
                if (inner.nodeType == NodeType::NEG)
                    return inner.children[0];
                break;
            }

            // ── ABS ──────────────────────────────────────────────────────────
            case NodeType::ABS: {
                const ExprNode& inner = node.children[0];
                // ABS(ABS(x)) → ABS(x)
                if (inner.nodeType == NodeType::ABS)
                    return inner;
                // ABS(NEG(x)) → ABS(x)
                if (inner.nodeType == NodeType::NEG) {
                    ExprNode simplified;
                    simplified.nodeType = NodeType::ABS;
                    simplified.children.push_back(inner.children[0]);
                    return simplified;
                }
                break;
            }

            // ── INV ──────────────────────────────────────────────────────────
            case NodeType::INV: {
                const ExprNode& inner = node.children[0];
                // INV(INV(x)) → x
                if (inner.nodeType == NodeType::INV)
                    return inner.children[0];
                break;
            }

            // ── MAX ──────────────────────────────────────────────────────────
            case NodeType::MAX_OP: {
                // MAX(x, x) → x
                if (structurallyEqual(node.children[0], node.children[1]))
                    return node.children[0];
                break;
            }

            // ── MIN ──────────────────────────────────────────────────────────
            case NodeType::MIN_OP: {
                // MIN(x, x) → x
                if (structurallyEqual(node.children[0], node.children[1]))
                    return node.children[0];
                break;
            }

            // ── SUB ──────────────────────────────────────────────────────────
            case NodeType::SUB: {
                // SUB(x, x) → CONST(0)
                if (structurallyEqual(node.children[0], node.children[1])) {
                    ExprNode zero;
                    zero.nodeType   = NodeType::CONST;
                    zero.constValue = 0.0;
                    return zero;
                }
                break;
            }

            // ── DIV ──────────────────────────────────────────────────────────
            case NodeType::DIV: {
                // DIV(x, x) → CONST(1)
                if (structurallyEqual(node.children[0], node.children[1])) {
                    ExprNode one;
                    one.nodeType   = NodeType::CONST;
                    one.constValue = 1.0;
                    return one;
                }
                break;
            }

            default:
                break;
        }

        return node;
    }

    /// @brief Count the number of nodes in the subtree.
    /// @return The number of nodes.
    int size() const {
        int s = 1;
        for (const auto& c : children) s += c.size();
        return s;
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