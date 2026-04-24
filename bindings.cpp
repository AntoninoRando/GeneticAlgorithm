#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // needed for vector conversion

#include "scheduler/decoder.h"
#include "scheduler/expr_tree.h"
#include "scheduler/registry.h"
#include "scheduler/tree_builder.h"
#include "scheduler/tree_operators.h"
#include "scheduler/types.h"

#define const
#include "scheduler/scheduler_gp.h"
#undef const

namespace py = pybind11;

PYBIND11_MODULE(scheduler, m) {
    m.doc() = "Python bindings for the Scheduler GP module";

    py::enum_<NodeType>(m, "NodeType", "Enum representing the type of operation or value in an expression tree node")
        .value("ADD", NodeType::ADD, "Addition operator")
        .value("SUB", NodeType::SUB, "Subtraction operator")
        .value("MUL", NodeType::MUL, "Multiplication operator")
        .value("DIV", NodeType::DIV, "Division operator")
        .value("MAX_OP", NodeType::MAX_OP, "Maximum of two values")
        .value("MIN_OP", NodeType::MIN_OP, "Minimum of two values")
        .value("NEG", NodeType::NEG, "Negation operator")
        .value("ABS", NodeType::ABS, "Absolute value operator")
        .value("INV", NodeType::INV, "Inverse value operator")
        .value("TERMINAL", NodeType::TERMINAL, "Terminal node (variable)")
        .value("CONST", NodeType::CONST, "Constant value node");

    py::class_<ExprNode>(m, "ExprNode", "A node in the expression tree used by the GP individual")
        .def_readwrite("nodeType", &ExprNode::nodeType, "Type of the node (e.g., ADD, CONST, TERMINAL)")
        .def_readwrite("terminalIndex", &ExprNode::terminalIndex, "Index of the terminal if the node is a TERMINAL")
        .def_readwrite("constValue", &ExprNode::constValue, "Constant value if the node is a CONST")
        .def_readwrite("children", &ExprNode::children, "List of child nodes")
        .def("size", &ExprNode::size, "Get the size of the subtree rooted at this node");

    py::class_<Satellite>(m, "Satellite", "Represents a computing satellite resource")
        .def(py::init([](int id, std::string name) {
            Satellite s;
            s.id   = id;
            s.name = std::move(name);
            return s;
        }), py::arg("id"), py::arg("name"), "Initialize a new satellite with an ID and name")
        .def_readwrite("id",                 &Satellite::id, "Unique identifier of the satellite")
        .def_readwrite("name",               &Satellite::name, "Name of the satellite")
        .def_readwrite("listeningDome",      &Satellite::listeningDome, "The listening dome availability state")
        .def_readwrite("activeTasks",        &Satellite::activeTasks, "Number of active tasks running on the satellite")
        .def_readwrite("RemainingEnergy",    &Satellite::RemainingEnergy, "Remaining energy available for computation")
        .def_readwrite("ComputingLoad",      &Satellite::ComputingLoad, "Current computing load")
        .def_readwrite("ComputingCapability",&Satellite::ComputingCapability, "Max computing capability of the satellite");

    // Job constructor argument order matches main.py: (id, name, durationMinutes, dueMinute, priority).
    py::class_<Job>(m, "Job", "Represents a job/task to be scheduled")
        .def(py::init([](int id, std::string name, int durationMinutes, int dueMinute) {
            Job j;
            j.id              = id;
            j.name            = std::move(name);
            j.durationMinutes = durationMinutes;
            j.dueMinute       = dueMinute;
            j.priority        = 1.0;  // Default priority
            return j;
        }), py::arg("id"), py::arg("name"), py::arg("durationMinutes"), py::arg("dueMinute"), "Initialize a new Job")
        .def_readwrite("id",                &Job::id, "Unique identifier of the job")
        .def_readwrite("name",              &Job::name, "Name of the job")
        .def_readwrite("durationMinutes",   &Job::durationMinutes, "Required duration to complete in minutes")
        .def_readwrite("dueMinute",         &Job::dueMinute, "The minute by which the job is due")
        .def_readwrite("priority",          &Job::priority, "Priority level of the job")
        .def_readwrite("arrivalTime",       &Job::arrivalTime, "When the job originally arrived")
        .def_readwrite("taskSize",          &Job::taskSize, "Size or complexity of the task")
        .def_readwrite("initialDeadline",   &Job::initialDeadline, "Initial established deadline")
        .def_readwrite("remainingDeadline", &Job::remainingDeadline, "Remaining time before the deadline is missed");

    py::class_<GPIndividual>(m, "Individual", "Represents a single candidate solution in the Genetic Programming population")
           .def_readwrite("id", &GPIndividual::id, "Unique identifier of the individual")
           .def_readwrite("tree", &GPIndividual::tree, "The individual's expression tree root node")
           .def_readwrite("fitness", &GPIndividual::fitness, "Evaluated fitness score");

    py::class_<SchedulerGP>(m, "SchedulerGP", "Genetic Programming Scheduler Manager")
        .def(py::init<std::vector<Satellite>, std::vector<Job>>(), py::arg("satellites"), py::arg("jobs"), "Initialize GP solver with targets")
           .def("initialize", &SchedulerGP::initialize,
               py::arg("populationSize") = 200,
               py::arg("crossoverRate") = 0.80,
               py::arg("mutationRate") = 0.08,
               py::arg("elitismCount") = 2,
               "Initialize the GP population with parameters")
           .def("solveNextGeneration", &SchedulerGP::solveNextGeneration, "Evaluate the current population and evolve to the next generation")
           .def("getPopulation", &SchedulerGP::getPopulation,
               py::return_value_policy::reference_internal, "Get the current population of individuals")
           .def("printSchedule", &SchedulerGP::printSchedule, "Print the best schedule found so far");
}