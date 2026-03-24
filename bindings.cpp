#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // needed for vector conversion
#include "scheduler.h"     

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

    py::class_<Individual>(m, "Individual")
        .def_readonly("fitness", &Individual::fitness)
        .def_readonly("scheduledJobs", &Individual::scheduledJobs)
        .def_readonly("servedPriority", &Individual::servedPriority)
        .def_readonly("conflicts", &Individual::conflicts)
        .def_readonly("lateness", &Individual::lateness)
        .def_readonly("genes", &Individual::genes);

    py::class_<SchedulerGA>(m, "SchedulerGA")
        .def(py::init<std::vector<Satellite>, std::vector<Job>>())
        .def("solve", &SchedulerGA::solve)
        .def("printSchedule", &SchedulerGA::printSchedule);
}