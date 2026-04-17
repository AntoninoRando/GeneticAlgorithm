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
    py::enum_<NodeType>(m, "NodeType")
        .value("ADD", NodeType::ADD)
        .value("SUB", NodeType::SUB)
        .value("MUL", NodeType::MUL)
        .value("DIV", NodeType::DIV)
        .value("MAX_OP", NodeType::MAX_OP)
        .value("MIN_OP", NodeType::MIN_OP)
        .value("NEG", NodeType::NEG)
        .value("ABS", NodeType::ABS)
        .value("INV", NodeType::INV)
        .value("TERMINAL", NodeType::TERMINAL)
        .value("CONST", NodeType::CONST);

    py::class_<ExprNode>(m, "ExprNode")
        .def_readwrite("nodeType", &ExprNode::nodeType)
        .def_readwrite("terminalIndex", &ExprNode::terminalIndex)
        .def_readwrite("constValue", &ExprNode::constValue)
        .def_readwrite("children", &ExprNode::children)
        .def("size", &ExprNode::size);

    py::class_<Satellite>(m, "Satellite")
        .def(py::init([](int id, std::string name) {
            Satellite s;
            s.id   = id;
            s.name = std::move(name);
            return s;
        }))
        .def_readwrite("id",                 &Satellite::id)
        .def_readwrite("name",               &Satellite::name)
        .def_readwrite("listeningDome",      &Satellite::listeningDome)
        .def_readwrite("activeTasks",        &Satellite::activeTasks)
        .def_readwrite("RemainingEnergy",    &Satellite::RemainingEnergy)
        .def_readwrite("ComputingLoad",      &Satellite::ComputingLoad)
        .def_readwrite("ComputingCapability",&Satellite::ComputingCapability);

    // Job constructor argument order matches main.py: (id, name, durationMinutes, dueMinute, priority).
    py::class_<Job>(m, "Job")
        .def(py::init([](int id, std::string name, int durationMinutes, int dueMinute) {
            Job j;
            j.id              = id;
            j.name            = std::move(name);
            j.durationMinutes = durationMinutes;
            j.dueMinute       = dueMinute;
            j.priority        = 1.0;  // Default priority
            return j;
        }))
        .def_readwrite("id",                &Job::id)
        .def_readwrite("name",              &Job::name)
        .def_readwrite("durationMinutes",   &Job::durationMinutes)
        .def_readwrite("dueMinute",         &Job::dueMinute)
        .def_readwrite("priority",          &Job::priority)
        .def_readwrite("arrivalTime",       &Job::arrivalTime)
        .def_readwrite("taskSize",          &Job::taskSize)
        .def_readwrite("initialDeadline",   &Job::initialDeadline)
        .def_readwrite("remainingDeadline", &Job::remainingDeadline);

    py::class_<GPIndividual>(m, "Individual")
           .def_readwrite("id", &GPIndividual::id)
           .def_readwrite("tree", &GPIndividual::tree)
           .def_readwrite("fitness", &GPIndividual::fitness);

    py::class_<SchedulerGP>(m, "SchedulerGP")
        .def(py::init<std::vector<Satellite>, std::vector<Job>>())
           .def("initialize", &SchedulerGP::initialize,
               py::arg("populationSize") = 200,
               py::arg("crossoverRate") = 0.80,
               py::arg("mutationRate") = 0.08,
               py::arg("elitismCount") = 2)
           .def("solveNextGeneration", &SchedulerGP::solveNextGeneration)
           .def("getPopulation", &SchedulerGP::getPopulation,
               py::return_value_policy::reference_internal)
           .def("printSchedule", &SchedulerGP::printSchedule);
}