# Project

The project is made of two parts
1. C++ source code that directly build into a python library (`scheduler`);
2. Python code that test the library.

The `scheduler` library offers an implementation of a **genetic programming algorithms** which workns on a simulation of **computational jobs** and **LEO satellites** to  evolves a _population_ of **scheduling functions**.

**Goal.** The goal of the project is to find, throguh a genetic algorithm, a scheduling formula capable of minimizing the average response time and energy consumption while maximizing the number of scheduled jobs.

## Scheduling Formula and Fitness Function
A scheduling formula takes as arguments a satellite and a job and return a value between 0 and 1 which describe how much that satellite is a match for that job.

During a generation of the genetic algorithm, each formula is tested against the simulation in the following way: the formula is evaluated between each job and satellites, then a satellite to schedule the job on is selected based on the result.

After each job in a simulation snapshot is being scheduled through a specific formula, that formula is evaluated by measuring:
- how many jobs have been completed;
- the average time to complete a job;
- the average energy consumption to complete a job.

These tree metrics are put together in a single value called the **fitness** of the scheduling formula. The genetic algorithm uses this fitness to carry the best formulas across the generations.