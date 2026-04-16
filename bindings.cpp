#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // needed for vector conversion
#include "scheduler/scheduler_gp.h"

namespace py = pybind11;

PYBIND11_MODULE(scheduler, m) {
    py::class_<Opportunity>(m, "Opportunity")
        .def(py::init<int, int, int>())
        .def_readwrite("satelliteId", &Opportunity::satelliteId)
        .def_readwrite("startMinute", &Opportunity::startMinute)
        .def_readwrite("endMinute", &Opportunity::endMinute);

    py::class_<Satellite>(m, "Satellite")
        .def(py::init<int, std::string>())
        .def_readwrite("id", &Satellite::id)
        .def_readwrite("name", &Satellite::name)
        .def_readwrite("listeningDome", &Satellite::listeningDome)
        .def_readwrite("activeTasks", &Satellite::activeTasks)
        .def_readwrite("RemainingEnergy", &Satellite::RemainingEnergy)
        .def_readwrite("ComputingLoad", &Satellite::ComputingLoad)
        .def_readwrite("ComputingCapability", &Satellite::ComputingCapability);

    py::class_<Job>(m, "Job")
        .def(py::init<int, std::string, int, int, double, std::vector<Opportunity>>())
        .def_readwrite("id", &Job::id)
        .def_readwrite("name", &Job::name)
        .def_readwrite("durationMinutes", &Job::durationMinutes)
        .def_readwrite("dueMinute", &Job::dueMinute)
        .def_readwrite("priority", &Job::priority)
        .def_readwrite("opportunities", &Job::opportunities)
        .def_readwrite("arrivalTime", &Job::arrivalTime)
        .def_readwrite("taskSize", &Job::taskSize)
        .def_readwrite("initialDeadline", &Job::initialDeadline)
        .def_readwrite("remainingDeadline", &Job::remainingDeadline);

    py::class_<GPIndividual>(m, "Individual")
        .def_property_readonly("fitness", [](const GPIndividual& ind) { return ind.result.fitness; })
        .def_property_readonly("scheduledJobs", [](const GPIndividual& ind) { return ind.result.scheduledJobs; })
        .def_property_readonly("servedPriority", [](const GPIndividual& ind) { return ind.result.servedPriority; })
        .def_property_readonly("conflicts", [](const GPIndividual& ind) { return ind.result.conflicts; })
        .def_property_readonly("lateness", [](const GPIndividual& ind) { return ind.result.lateness; });

    py::class_<SchedulerGP>(m, "SchedulerGA")
        .def(py::init<std::vector<Satellite>, std::vector<Job>>())
        .def("solve", &SchedulerGP::solve)
        .def("printSchedule", &SchedulerGP::printSchedule);
}