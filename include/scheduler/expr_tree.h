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