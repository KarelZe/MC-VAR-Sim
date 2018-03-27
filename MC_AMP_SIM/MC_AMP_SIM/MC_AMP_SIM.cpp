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

typedef std::vector<double> path;
typedef std::vector<std::vector<double>> paths;
typedef std::vector<double> pathSimplified;
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
void calculatePaths(paths &paths) {
	// scale drift to timestep
	const double dailyDrift = RETURN / TRADING_DAYS;
	// scale volatility to timestep, note that volatility scales with the square root of time
	const double dailyVolatility = VOLATILITY / sqrt(TRADING_DAYS);
	const double expectedDailyDrift = dailyDrift - 0.5 * dailyVolatility * dailyVolatility;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> normal(0, 1);

	for(path &path : paths){
		path.at(0) = INITIAL_VALUE;
		for (unsigned int i(1); i < path.size(); i++) {
			double shock = normal(generator) * dailyVolatility;
			// calculate log return from drift and shock. Log return is generally prefered over absolute returns etc.
			double logReturn = expectedDailyDrift + shock;
			// calculate the price after one day based on the initial value and the return
			path.at(i) = std::max(path.at(i-1) * exp(logReturn), 0.0);
		}
	}
}
/* second approach for calculating value at risk. 
Calculates the value at risk for one day using a MonteCarlo method*/
void calculatePathsSimplified(pathSimplified &paths) {
	// scale drift to timestep
	const double dailyDrift = RETURN / TRADING_DAYS;
	// scale volatility to timestep, note that volatility scales with the square root of time
	const double dailyVolatility = VOLATILITY / sqrt(TRADING_DAYS);
	const double expectedDailyDrift = dailyDrift - 0.5 * dailyVolatility * dailyVolatility;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> normal(0, 1);

	for (unsigned int i(0); i < paths.size(); i++) {
		double shock = normal(generator) * dailyVolatility;
		// calculate log return from drift and shock. Log return is generally prefered over absolute returns etc.
		double logReturn = expectedDailyDrift + shock;
		// calculate the price after one day based on the initial value and the return
		paths.at(i) = std::max(INITIAL_VALUE * exp(logReturn), 0.0);
	}
}

/*
	function to scale daily value at risk to holding period
*/
void calculateValueAtRiskSimplified(pathSimplified &paths) {
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
void calculateValueAtRisk(paths &paths) {
	// extract end values from random paths
	path endValues;
	for (path p : paths)
		endValues.push_back(p.at(p.size()-1));

	int rank = static_cast<int> ((1.0 - CONFIDENCE_LEVEL) * NUMBER_OF_PATHS);
	std::nth_element(endValues.begin(), endValues.begin() + rank, endValues.end());
	std::cout << std::fixed << std::setprecision(5);
	std::cout << "(c) value at rank: " << endValues.at(rank) << std::endl;
	double valueAtRisk = endValues.at(rank) - INITIAL_VALUE;
	std::cout << "(c) value at risk at " << HOLDING_PERIOD << " days with "<< CONFIDENCE_LEVEL * 100<<" % confidence: "<< valueAtRisk << " (with - being risk and + being chance)" << std::endl;
}

/* helper function to write content of a two dimensional vector to
semicolon seperated csv file used to export path data for visualization.
*/
void writeToCSV(paths paths) {
	file.open("measures.csv", std::ios::out | std::ios::app);
	int pathId(0);
	for (path path : paths) {
		file << pathId << ";";
		for (double price : path) {
			file << price << ";";
		}
		file << std::endl;
		pathId++;
	}
	file.close();
}


int main()
{
	// complex method
	// generate multidimensional vector with size [NUMBER_OF_PATHS * HOLDING_PERIOD] = {0,0,0,0...}
	paths paths(NUMBER_OF_PATHS, path(HOLDING_PERIOD, 0));
	// calculate random paths using geometric brownian motion
	calculatePaths(paths);
	// calculate value at risk from paths and scale to desired confidence level and holding period
	calculateValueAtRisk(paths);
	// write paths to csv file for plotting
	// writeToCSV(paths);

	// simplified method
	pathSimplified pathsSimplified(NUMBER_OF_PATHS);
	// calculate paths for one day
	calculatePathsSimplified(pathsSimplified);
	// derive value at risk from paths and scale to horizon
	calculateValueAtRiskSimplified(pathsSimplified);

    return 0;
}

