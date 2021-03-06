// MC_AMP_SIM.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include<vector>
#include<algorithm>
#include<iostream>
#include<iomanip>
#include<random>
#include<fstream>
#include<iostream>
#include<cmath>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;

typedef std::vector<double> path;
typedef std::vector<std::vector<double>> paths;
typedef std::vector<double> path_simplified;
typedef std::chrono::steady_clock the_clock;
std::ofstream file;

const unsigned int NUMBER_OF_PATHS = 100'000;
const double VOLATILITY = 0.04;
const double INITIAL_VALUE = 10;
const double RETURN = 0.05;
const double CONFIDENCE_LEVEL = 0.99;
const unsigned int HOLDING_PERIOD = 300;
const unsigned int TRADING_DAYS = 250;

/* generate random paths using geometric brownian motion
*/
void calculate_paths(paths &paths) {
	// scale drift to timestep
	const double daily_drift = RETURN / TRADING_DAYS;
	// scale volatility to timestep, note that volatility scales with the square root of time
	const double daily_volatility = VOLATILITY / sqrt(TRADING_DAYS);
	const double expected_daily_drift = daily_drift - 0.5 * daily_volatility * daily_volatility;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> normal(0, 1);

	for(path &path : paths){
		path.at(0) = INITIAL_VALUE;
		for (unsigned int i(1); i < path.size(); i++) {
			double shock = normal(generator) * daily_volatility;
			// calculate log return from drift and shock. Log return is generally prefered over absolute returns etc.
			double logReturn = expected_daily_drift + shock;
			// calculate the price after one day based on the initial value and the return
			path.at(i) = std::max(path.at(i-1) * exp(logReturn), 0.0);
		}
	}
}
/* second approach for calculating value at risk. 
Calculates the value at risk for one day using a MonteCarlo method*/
void calculate_paths_simplified(path_simplified &paths) {
	// scale drift to timestep
	const double daily_drift = RETURN / TRADING_DAYS;
	// scale volatility to timestep, note that volatility scales with the square root of time
	const double daily_volatility = VOLATILITY / sqrt(TRADING_DAYS);
	const double expected_daily_drift = daily_drift - 0.5 * daily_volatility * daily_volatility;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> normal(0, 1);

	for (unsigned int i(0); i < paths.size(); i++) {
		double shock = normal(generator) * daily_volatility;
		// calculate log return from drift and shock. Log return is generally prefered over absolute returns etc.
		double logReturn = expected_daily_drift + shock;
		// calculate the price after one day based on the initial value and the return
		paths.at(i) = std::max(INITIAL_VALUE * exp(logReturn), 0.0);
	}
}

/*
	function to scale daily value at risk to holding period
*/
void calculate_value_at_risk_simplified(path_simplified &paths) {
	int rank = static_cast<int> ((1.0 - CONFIDENCE_LEVEL) * NUMBER_OF_PATHS);
	std::nth_element(paths.begin(), paths.begin() + rank, paths.end());
	std::cout << std::fixed << std::setprecision(5);
	std::cout << "(s) value at rank: " << paths.at(rank) << std::endl;
	double valueAtRiskUnscaled = paths.at(rank) - INITIAL_VALUE;
	std::cout << "(s) daily value at risk: " << valueAtRiskUnscaled << " (with - being risk and + being chance)" << std::endl;
	double valueAtRiskScaled = valueAtRiskUnscaled * sqrt(HOLDING_PERIOD);
	std::cout << "(s) value at risk at " << HOLDING_PERIOD << " days with " << CONFIDENCE_LEVEL * 100 << " % confidence: " << valueAtRiskScaled << " (with - being risk and + being chance)" << std::endl;
}

/*
	function to extract value at risk from random paths.
*/
void calculate_value_at_risk(paths &paths) {
	// extract end values from random paths
	path end_values;
	for (path p : paths)
		end_values.push_back(p.at(p.size()-1));

	int rank = 0;
	std::nth_element(end_values.begin(), end_values.begin() + rank, end_values.end());

	//int rank = static_cast<int> ((1.0 - CONFIDENCE_LEVEL) * NUMBER_OF_PATHS);
	//std::cout << std::fixed << std::setprecision(5);
	//std::cout << "(c) value at rank: " << endValues.at(rank) << std::endl;
	//std::cout << "(c) value at risk at " << HOLDING_PERIOD << " days with "<< CONFIDENCE_LEVEL * 100<<" % confidence: "<< valueAtRisk << " (with - being risk and + being chance)" << std::endl;
}

/* helper function to write content of a two dimensional vector to
semicolon seperated csv file used to export path data for visualization.
*/
void write_to_csv(paths paths) {
	file.open("measures.csv", std::ios::out | std::ios::app);
	int path_id(0);
	for (path path : paths) {
		file << path_id << ";";
		for (double price : path) {
			file << price << ";";
		}
		file << std::endl;
		path_id++;
	}
	file.close();
}


int main()
{
	file.open("measures_cpu.csv", std::ios::out);

	for (auto ps(1024); ps <= 524'288; ps *= 2) {
		
		file << ps << ",";
		the_clock::time_point start = the_clock::now();
		// generate multidimensional vector with size [NUMBER_OF_PATHS * HOLDING_PERIOD] = {0,0,0,0...}
		paths paths(ps, path(HOLDING_PERIOD, 0));
		// calculate random paths using geometric brownian motion
		calculate_paths(paths);
		// calculate value at risk from paths and scale to desired confidence level and holding period
		calculate_value_at_risk(paths);
	
		the_clock::time_point end = the_clock::now();
		auto time_taken = duration_cast<milliseconds>(end - start).count();
		file << time_taken << std::endl;
	}
	// close file stream
	file.close();

	//writeToCSV(paths);

	// simplified method
	// pathSimplified pathsSimplified(NUMBER_OF_PATHS);
	// calculate paths for one day
	// calculatePathsSimplified(pathsSimplified);
	// derive value at risk from paths and scale to horizon
	// calculateValueAtRiskSimplified(pathsSimplified);

    return 0;
}

