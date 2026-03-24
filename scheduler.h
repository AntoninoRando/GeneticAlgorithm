#pragma once

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;



#pragma region STRUCTS ---------------------------------------------------------
struct Opportunity {
	int satelliteId;
	int startMinute;
	int endMinute;
};

struct Job {
	int id;
	string name;
	int durationMinutes;
	int dueMinute;
	double priority;
	vector<Opportunity> opportunities;

	std::time_t arrivalTime = 0;
	double      taskSize;
	double      initialDeadline;
	double      remainingDeadline;
};

struct Satellite {
	int id;
	string name;
	
	vector<Satellite> listeningDome;   // Set of satellites that can listen to this satellite's communication.

	int    activeTasks;            
	double RemainingEnergy;
	double ComputingLoad;
	double ComputingCapability;
};

struct Individual {
	// gene[i] = chosen opportunity index for job i, or -1 to skip the job.
	vector<int> genes;
	double fitness = -numeric_limits<double>::infinity();
	double servedPriority = 0.0;
	int scheduledJobs = 0;
	int conflicts = 0;
	int lateness = 0;
};
#pragma endregion --------------------------------------------------------------



#pragma region GENETIC ALGORITHM -----------------------------------------------
class SchedulerGA {
public:
	#pragma region CONSTRUCTOR -------------------------------------------------
	SchedulerGA(vector<Satellite> satellites, vector<Job> jobs)
		: satellites_(move(satellites)), 
		  jobs_(move(jobs)), 
		  rng_(random_device{}()) { }
	#pragma endregion ----------------------------------------------------------



	Individual solve(int populationSize, int generations, double crossoverRate, double mutationRate) {
		vector<Individual> population;
		population.reserve(populationSize);

		for (int i = 0; i < populationSize; ++i) {
			population.push_back(randomIndividual());
			evaluate(population.back());
		}

		Individual best = *max_element(population.begin(), population.end(), byFitness);

		for (int generation = 0; generation < generations; ++generation) {
			vector<Individual> nextPopulation;
			nextPopulation.reserve(populationSize);

			// Elitism: keep best two individuals.
			sort(population.begin(), population.end(), byFitness);
			nextPopulation.push_back(population.back());
			if (population.size() > 1) {
				nextPopulation.push_back(population[population.size() - 2]);
			}

			while (static_cast<int>(nextPopulation.size()) < populationSize) {
				const Individual& parentA = tournamentSelect(population, 3);
				const Individual& parentB = tournamentSelect(population, 3);

				Individual childA = parentA;
				Individual childB = parentB;

				if (randomReal(0.0, 1.0) < crossoverRate) {
					tie(childA, childB) = crossover(parentA, parentB);
				}

				mutate(childA, mutationRate);
				evaluate(childA);
				nextPopulation.push_back(childA);

				if (static_cast<int>(nextPopulation.size()) < populationSize) {
					mutate(childB, mutationRate);
					evaluate(childB);
					nextPopulation.push_back(childB);
				}
			}

			population = move(nextPopulation);
			const Individual& generationBest = *max_element(population.begin(), population.end(), byFitness);
			if (generationBest.fitness > best.fitness) {
				best = generationBest;
			}

			if ((generation + 1) % 50 == 0 || generation == generations - 1) {
				cout << "Generation " << setw(4) << generation + 1
					 << " | Best fitness: " << fixed << setprecision(2) << best.fitness
					 << " | Scheduled jobs: " << best.scheduledJobs
					 << " | Conflicts: " << best.conflicts
					 << " | Lateness: " << best.lateness << "\n";
			}
		}

		return best;
	}

	void printSchedule(const Individual& best) const {
		cout << "\n=== Best Schedule Summary ===\n";
		cout << "Fitness: " << fixed << setprecision(2) << best.fitness << "\n";
		cout << "Scheduled jobs: " << best.scheduledJobs << " / " << jobs_.size() << "\n";
		cout << "Served priority: " << best.servedPriority << "\n";
		cout << "Conflicts: " << best.conflicts << "\n";
		cout << "Total lateness (minutes): " << best.lateness << "\n\n";

		cout << left << setw(12) << "Job"
			 << setw(12) << "Satellite"
			 << setw(14) << "Start(min)"
			 << setw(14) << "End(min)"
			 << setw(10) << "Due(min)"
			 << setw(10) << "Late" << "\n";
		cout << string(72, '-') << "\n";

		for (size_t j = 0; j < jobs_.size(); ++j) {
			const int chosen = best.genes[j];
			if (chosen < 0) {
				cout << left << setw(12) << jobs_[j].name
					 << setw(12) << "-"
					 << setw(14) << "-"
					 << setw(14) << "-"
					 << setw(10) << jobs_[j].dueMinute
					 << setw(10) << "-" << "\n";
				continue;
			}

			const Opportunity& op = jobs_[j].opportunities[chosen];
			const int late = max(0, op.endMinute - jobs_[j].dueMinute);
			cout << left << setw(12) << jobs_[j].name
				 << setw(12) << satellites_[op.satelliteId].name
				 << setw(14) << op.startMinute
				 << setw(14) << op.endMinute
				 << setw(10) << jobs_[j].dueMinute
				 << setw(10) << late << "\n";
		}
	}

private:
	#pragma region FIELDS ------------------------------------------------------
	vector<Satellite> satellites_;
	vector<Job> jobs_;
	mt19937 rng_;
	#pragma endregion ----------------------------------------------------------



	#pragma region STATIC FUNCTIONS --------------------------------------------
	/**
	 * @brief An `Individual` comparator that compare them using their fitness
	 * score. 
	 */
	static bool byFitness(const Individual& a, const Individual& b) {
		return a.fitness < b.fitness;
	}
	#pragma endregion ----------------------------------------------------------



	#pragma region UTILITIES ---------------------------------------------------
	/**
	 * @brief Returns a random integer in the given interval.
	 */
	int randomInt(int minVal, int maxVal) {
		uniform_int_distribution<int> dist{minVal, maxVal};
		return dist(rng_);
	}

	/**
	 * @brief Returns a random double in the given interval.
	 */
	double randomReal(double minVal, double maxVal) {
		uniform_real_distribution<double> dist{minVal, maxVal};
		return dist(rng_);
	}

	Individual randomIndividual() {
		Individual ind;
		ind.genes.resize(jobs_.size(), -1);

		for (size_t j = 0; j < jobs_.size(); ++j) {
			// 20% probability to leave the job unscheduled.
			if (jobs_[j].opportunities.empty() || randomReal(0.0, 1.0) < 0.2) {
				ind.genes[j] = -1;
			} else {
				ind.genes[j] = randomInt(0, static_cast<int>(jobs_[j].opportunities.size()) - 1);
			}
		}
		return ind;
	}
	#pragma endregion ----------------------------------------------------------



	/**
     * @brief Computes the fitness of an individual in-place.
	 * @param ind The individual to be evaluated.
	 */
	void evaluate(Individual& ind) const {
		const int minGap = 3; // Minimum reconfiguration gap between jobs on same satellite.

		ind.scheduledJobs = 0;
		ind.servedPriority = 0.0;
		ind.conflicts = 0;
		ind.lateness = 0;

		vector<vector<pair<int, int>>> satIntervals(satellites_.size());

		for (size_t j = 0; j < jobs_.size(); ++j) {
			const int chosen = ind.genes[j];
			if (chosen < 0) {
				continue;
			}

			if (chosen >= static_cast<int>(jobs_[j].opportunities.size())) {
				// Invalid gene gets harshly penalized.
				ind.conflicts += 5;
				continue;
			}

			const Opportunity& op = jobs_[j].opportunities[chosen];
			ind.scheduledJobs++;
			ind.servedPriority += jobs_[j].priority;
			ind.lateness += max(0, op.endMinute - jobs_[j].dueMinute);

			satIntervals[op.satelliteId].push_back({op.startMinute, op.endMinute});
		}

		// Count overlap and gap conflicts per satellite.
		for (auto& intervals : satIntervals) {
			sort(intervals.begin(), intervals.end());
			for (size_t i = 1; i < intervals.size(); ++i) {
				const int prevEnd = intervals[i - 1].second;
				const int curStart = intervals[i].first;
				if (curStart < prevEnd) {
					ind.conflicts += 2;
				} else if (curStart - prevEnd < minGap) {
					ind.conflicts += 1;
				}
			}
		}

		const int unscheduledJobs = static_cast<int>(jobs_.size()) - ind.scheduledJobs;

		// Maximize served priority and completed jobs while penalizing conflicts and lateness.
		ind.fitness = 40.0 * ind.servedPriority
			+ 6.0 * ind.scheduledJobs
			- 80.0 * ind.conflicts
			- 0.6 * ind.lateness
			- 12.0 * unscheduledJobs;
	}

	const Individual& tournamentSelect(const vector<Individual>& population, int k) {
		int bestIndex = randomInt(0, static_cast<int>(population.size()) - 1);
		for (int i = 1; i < k; ++i) {
			int candidate = randomInt(0, static_cast<int>(population.size()) - 1);
			if (population[candidate].fitness > population[bestIndex].fitness) {
				bestIndex = candidate;
			}
		}
		return population[bestIndex];
	}

	pair<Individual, Individual> crossover(const Individual& a, const Individual& b) {
		Individual childA = a;
		Individual childB = b;

		if (a.genes.empty()) {
			return {childA, childB};
		}

		const int cut = randomInt(1, static_cast<int>(a.genes.size()) - 1);
		for (int i = cut; i < static_cast<int>(a.genes.size()); ++i) {
			childA.genes[i] = b.genes[i];
			childB.genes[i] = a.genes[i];
		}

		return {childA, childB};
	}

	void mutate(Individual& ind, double mutationRate) {
		for (size_t j = 0; j < ind.genes.size(); ++j) {
			if (randomReal(0.0, 1.0) > mutationRate) {
				continue;
			}

			if (jobs_[j].opportunities.empty()) {
				ind.genes[j] = -1;
				continue;
			}

			// Mutation can switch opportunity or drop the job.
			if (randomReal(0.0, 1.0) < 0.15) {
				ind.genes[j] = -1;
			} else {
				ind.genes[j] = randomInt(0, static_cast<int>(jobs_[j].opportunities.size()) - 1);
			}
		}
	}
};
#pragma endregion --------------------------------------------------------------



#pragma region FUNCTIONS -------------------------------------------------------
static vector<Satellite> createSatellites() {
	return {
		{0, "SEO-A", {{1, "SEO-B", {}, 0, 0.0, 0.0, 0.0}, {2, "SEO-C", {}, 0, 0.0, 0.0, 0.0}}, 0, 100.0, 0.20, 18.0},
		{1, "SEO-B", {{0, "SEO-A", {}, 0, 0.0, 0.0, 0.0}, {2, "SEO-C", {}, 0, 0.0, 0.0, 0.0}}, 0, 98.0, 0.25, 16.5},
		{2, "SEO-C", {{0, "SEO-A", {}, 0, 0.0, 0.0, 0.0}, {1, "SEO-B", {}, 0, 0.0, 0.0, 0.0}}, 0, 95.0, 0.18, 19.0}
	};
}

static vector<Job> createJobs() {
	// Time unit: minute from mission start.
	const std::time_t missionStartEpoch = 1700000000;
	return {
		{0, "Target-01", 8, 100, 5.0, {{0, 30, 38}, {1, 42, 50}, {2, 55, 63}}, missionStartEpoch + 10 * 60, 10.0, 100.0, 100.0},
		{1, "Target-02", 10, 145, 7.0, {{0, 70, 80}, {1, 92, 102}, {2, 110, 120}}, missionStartEpoch + 55 * 60, 14.0, 145.0, 145.0},
		{2, "Target-03", 12, 190, 9.0, {{0, 125, 137}, {1, 140, 152}, {2, 162, 174}}, missionStartEpoch + 105 * 60, 17.0, 190.0, 190.0},
		{3, "Target-04", 9, 205, 4.0, {{0, 155, 164}, {1, 175, 184}, {2, 188, 197}}, missionStartEpoch + 138 * 60, 9.5, 205.0, 205.0},
		{4, "Target-05", 7, 230, 6.5, {{0, 180, 187}, {1, 198, 205}, {2, 216, 223}}, missionStartEpoch + 165 * 60, 11.0, 230.0, 230.0},
		{5, "Target-06", 11, 280, 8.5, {{0, 220, 231}, {1, 240, 251}, {2, 255, 266}}, missionStartEpoch + 205 * 60, 16.0, 280.0, 280.0},
		{6, "Target-07", 8, 315, 5.5, {{0, 260, 268}, {1, 272, 280}, {2, 295, 303}}, missionStartEpoch + 245 * 60, 10.5, 315.0, 315.0},
		{7, "Target-08", 10, 350, 7.5, {{0, 300, 310}, {1, 320, 330}, {2, 338, 348}}, missionStartEpoch + 285 * 60, 13.5, 350.0, 350.0},
		{8, "Target-09", 12, 390, 9.5, {{0, 335, 347}, {1, 355, 367}, {2, 370, 382}}, missionStartEpoch + 320 * 60, 18.0, 390.0, 390.0},
		{9, "Target-10", 9, 430, 6.0, {{0, 365, 374}, {1, 385, 394}, {2, 405, 414}}, missionStartEpoch + 350 * 60, 12.0, 430.0, 430.0}
	};
}
#pragma endregion --------------------------------------------------------------



#pragma region MAIN ------------------------------------------------------------
int main() {
	vector<Satellite> satellites = createSatellites();
	vector<Job> jobs = createJobs();

	SchedulerGA ga(satellites, jobs);

	const int populationSize = 160;
	const int generations = 500;
	const double crossoverRate = 0.85;
	const double mutationRate = 0.06;

	Individual best = ga.solve(populationSize, generations, crossoverRate, mutationRate);
	ga.printSchedule(best);

	return 0;
}
#pragma endregion --------------------------------------------------------------



#pragma region END -------------------------------------------------------------
#pragma endregion --------------------------------------------------------------