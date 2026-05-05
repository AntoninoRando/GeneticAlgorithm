#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // needed for vector conversion
#include <pybind11/stl_bind.h>   // ← add this include

#include "scheduler/expr_tree.h"
#include "scheduler/registry.h"
#include "scheduler/tree_builder.h"
#include "scheduler/tree_operators.h"
#include "scheduler/types.h"
#include "scheduler/fitness_evaluator.h"

#define const
#include "scheduler/scheduler_gp.h"
#undef const

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<GPIndividual>);
PYBIND11_MODULE(scheduler, m) {
    m.doc() = "Python bindings for the Scheduler GP module";

    // Inside PYBIND11_MODULE, before the SchedulerGP binding:
    py::bind_vector<std::vector<GPIndividual>>(m, "Population");

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
           .def("getPopulation",
                static_cast<vector<GPIndividual>& (SchedulerGP::*)()>(
                    &SchedulerGP::getPopulation),
                py::return_value_policy::reference_internal,
                "Get the current population of individuals (mutable reference).")
           .def("printSchedule", &SchedulerGP::printSchedule, "Print the best schedule found so far");
    
           // ── SimulationSnapshot ────────────────────────────────────────────────────
    py::class_<SimulationSnapshot>(m, "SimulationSnapshot",
        R"doc(
A self-contained, point-in-time snapshot of the simulation world.

The external application must construct and populate a snapshot before
calling :py:meth:`FitnessEvaluator.evaluate` for each GP generation.
All satellite and job objects are copied into the snapshot, so subsequent
changes to the originals have no effect on the evaluator.

Example
-------
.. code-block:: python

    snap = scheduler.SimulationSnapshot(
        currentMinute = 0.0,
        satellites    = my_satellites,
        jobs          = my_jobs,
        uniformEnergy = 2.5,   # watts per minute per satellite
    )
        )doc")

        .def(py::init<double, vector<Satellite>, vector<Job>, double>(),
             py::arg("currentMinute"),
             py::arg("satellites"),
             py::arg("jobs"),
             py::arg("uniformEnergy") = 1.0,
             R"doc(
Construct a snapshot.

Parameters
----------
currentMinute : float
    Simulation clock at the time of the snapshot, in minutes from the
    simulation epoch.  Jobs whose ``arrivalTime`` is greater than this
    value are treated as not yet available and are ignored by the decoder.
satellites : list[Satellite]
    Current state of every satellite in the constellation.
jobs : list[Job]
    Pool of jobs available for scheduling at this instant.
uniformEnergy : float, optional
    Energy consumed per minute of active computation, applied uniformly
    to every satellite.  Defaults to ``1.0``.  Pass ``0.0`` to use the
    per-satellite ``energyPerMinute`` vector instead.
             )doc")

        .def_readwrite("currentMinute", &SimulationSnapshot::currentMinute,
            "Simulation clock at snapshot time (minutes from epoch).")
        .def_readwrite("satellites", &SimulationSnapshot::satellites,
            "Current state of every satellite in the constellation.")
        .def_readwrite("jobs", &SimulationSnapshot::jobs,
            "Pool of jobs available for scheduling at this instant.")
        .def_readwrite("energyPerMinute", &SimulationSnapshot::energyPerMinute,
            R"doc(
Energy consumed per minute of computation, indexed by satellite position in
the ``satellites`` list.  When non-empty this overrides ``uniformEnergy``.
Must have the same length as ``satellites`` if provided.
            )doc");


    // ── FitnessEvaluator ──────────────────────────────────────────────────────
    py::class_<FitnessEvaluator>(m, "FitnessEvaluator",
        R"doc(
Evaluates a population of GP individuals by simulating a greedy schedule
with each individual's expression tree as the priority function.

After evaluation every :py:class:`Individual` in the population will have
its ``fitness`` field set to a scalar value that reflects how well the tree
minimised response time and energy consumption.

Fitness formula
---------------
Three raw objectives are collected for each individual:

* ``compRatio``  – fraction of available jobs that were successfully
  scheduled (higher is better).
* ``meanResp``   – average response time (completion - arrival) in minutes
  (lower is better).
* ``energy``     – total energy consumed across all satellites in watt-
  minutes (lower is better).

Each objective is normalised to ``[0, 1]`` across the current population
so that differences in scale cannot dominate the result.  The final fitness
is then:

.. math::

    f = w_{\\text{jobs}} \\cdot \\widehat{r}
      - w_{\\text{time}} \\cdot \\widehat{t}
      - w_{\\text{energy}} \\cdot \\widehat{e}

where :math:`\\hat{\\cdot}` denotes population-normalised values and the
default weights are ``(0.60, 0.25, 0.15)``.

Example
-------
.. code-block:: python

    from scheduler import SchedulerGP, FitnessEvaluator, SimulationSnapshot
    from scheduler import buildTerminalRegistry

    gp   = SchedulerGP(satellites, jobs)
    gp.initialize(populationSize=100)

    eval = FitnessEvaluator(buildTerminalRegistry())

    for generation in range(50):
        snap = SimulationSnapshot(current_minute, satellites, jobs)
        eval.evaluate(gp.getPopulation(), snap)  # writes fitness in place
        gp.solveNextGeneration()
        )doc")

        .def(py::init<vector<TerminalDef>>(),
             py::arg("registry"),
             R"doc(
Construct the evaluator.

Parameters
----------
registry : list[TerminalDef]
    Terminal registry produced by :py:func:`buildTerminalRegistry`.
             )doc")

        // ── Weights ───────────────────────────────────────────────────────
        .def_readwrite("weightJobs", &FitnessEvaluator::weightJobs,
            R"doc(
Weight applied to the completion-ratio reward term.  Default ``0.60``.
Increase this to make the GP prioritise scheduling as many jobs as possible.
            )doc")
        .def_readwrite("weightTime", &FitnessEvaluator::weightTime,
            R"doc(
Weight applied to the mean-response-time penalty.  Default ``0.25``.
Increase this to make the GP prioritise faster job turnaround.
            )doc")
        .def_readwrite("weightEnergy", &FitnessEvaluator::weightEnergy,
            R"doc(
Weight applied to the energy-consumption penalty.  Default ``0.15``.
Increase this to make the GP prefer energy-efficient satellite assignments.
            )doc")

        // ── Core method ───────────────────────────────────────────────────
        .def("evaluate",
             &FitnessEvaluator::evaluate,
             py::arg("population"),
             py::arg("snap"),
             R"doc(
Score every individual in *population* and write the result into each
individual's ``fitness`` field.

This is the only method the main GP loop needs to call each generation.

Parameters
----------
population : list[Individual]
    The current GP population, obtained via :py:meth:`SchedulerGP.getPopulation`.
    Each individual's ``fitness`` field is updated in place.
snap : SimulationSnapshot
    Read-only snapshot of the current simulation state.  The evaluator
    runs a fresh greedy simulation for every individual, so ``snap`` is
    never modified.

Notes
-----
Fitness values are population-relative: the same tree may receive a
different score in a different generation if the rest of the population
has changed.
             )doc")

        // ── Debug helper ──────────────────────────────────────────────────
        .def("evaluate_single",
             [](const FitnessEvaluator& self,
                const GPIndividual&       ind,
                const SimulationSnapshot& snap) -> py::tuple
             {
                 double cr, mr, en;
                 self.evaluateSingle(ind, snap, cr, mr, en);
                 return py::make_tuple(cr, mr, en);
             },
             py::arg("individual"),
             py::arg("snap"),
             R"doc(
Evaluate a single individual and return its raw (un-normalised) objectives.

Useful for inspecting the best individual after evolution has finished.

Parameters
----------
individual : Individual
    The GP individual to evaluate.
snap : SimulationSnapshot
    Simulation snapshot used for the evaluation.

Returns
-------
tuple[float, float, float]
    ``(comp_ratio, mean_response_time, total_energy)``

    * ``comp_ratio``          – fraction of available jobs scheduled ∈ [0, 1].
    * ``mean_response_time``  – average (completion - arrival) in minutes.
    * ``total_energy``        – total energy consumed (watt-minutes).
             )doc");

// ── TerminalDef ───────────────────────────────────────────────────────────────
py::class_<TerminalDef>(m, "TerminalDef",
    "A named terminal variable used in GP expression trees (e.g. 'job.priority').")
    .def_readonly("name", &TerminalDef::name,
        "Human-readable name of the terminal (e.g. ``'job.priority'``).");

// ── buildTerminalRegistry ─────────────────────────────────────────────────────
m.def("buildTerminalRegistry", &buildTerminalRegistry,
    R"doc(
Build and return the terminal registry used by the GP algorithm.

The registry maps human-readable names to field accessors that the expression
tree evaluator uses to read job and satellite features at runtime.  You must
pass the returned list to :py:class:`FitnessEvaluator` and, if you construct
a :py:class:`TreeBuilder` directly, to that as well.

Returns
-------
list[TerminalDef]
    Ordered list of terminal definitions.  The index of each entry corresponds
    to the ``terminalIndex`` stored in ``TERMINAL`` nodes of the expression tree.

Example
-------
.. code-block:: python

    from scheduler import buildTerminalRegistry, FitnessEvaluator

    registry = buildTerminalRegistry()
    evaluator = FitnessEvaluator(registry)
    )doc");
}