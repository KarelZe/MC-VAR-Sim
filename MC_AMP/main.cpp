#include <assert.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <time.h>
#include <string>
#include <amp_tinymt_rng.h>
#include <amp_math.h>


// Need to access the concurrency libraries 
using namespace concurrency;

// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;

// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_serial_clock;
typedef std::chrono::steady_clock the_amp_clock;

// Define variables needed for value at risk calculation
const int NUMBER_OF_PATHS = 1'048'576;
const float INITIAL_VALUE = 10.0f;
const float VOLATILITY = 0.04f;
const float EXPECTED_RETURN = 0.05f;
const float CONFIDENCE_LEVEL = 0.99f;
const unsigned int HOLDING_PERIOD = 300;
const unsigned int TRADING_DAYS = 250;
void report_accelerator(const accelerator a)
{
	const std::wstring bs[2] = { L"false", L"true" };
	std::wcout << ": " << a.description << " "
		<< std::endl << "       device_path                       = " << a.device_path
		<< std::endl << "       dedicated_memory                  = " << std::setprecision(4) << float(a.dedicated_memory) / (1024.0f * 1024.0f) << " Mb"
		<< std::endl << "       has_display                       = " << bs[a.has_display]
		<< std::endl << "       is_debug                          = " << bs[a.is_debug]
		<< std::endl << "       is_emulated                       = " << bs[a.is_emulated]
		<< std::endl << "       supports_double_precision         = " << bs[a.supports_double_precision]
		<< std::endl << "       supports_limited_double_precision = " << bs[a.supports_limited_double_precision]
		<< std::endl;
}
// List and select the accelerator to use. If default accelerator is reference implementation, a warning is thrown.
void list_accelerators()
{
	//get all accelerators available to us and store in a vector so we can extract details
	std::vector<accelerator> accls = accelerator::get_all();

	// iterates over all accelerators and print characteristics
	for (unsigned int i(0); i < accls.size(); i++)
	{
		accelerator a = accls[i];
		report_accelerator(a);
	}

	const accelerator acc = accelerator(accelerator::default_accelerator);
	std::wcout << " default acc = " << acc.description << std::endl;
	// todo: replace with assert?
	if (acc == accelerator(accelerator::direct3d_ref))
		std::wcout << "Running on very slow emulator! Only use this accelerator for debugging." << std::endl;
} // list_accelerators

 // query if AMP accelerator exists on hardware
void query_AMP_support()
{
	std::vector<accelerator> accls = accelerator::get_all();
	if (accls.empty())
	{
		std::cout << "No accelerators found that are compatible with C++ AMP" << std::endl;
	}
	else
	{
		std::cout << "Accelerators found that are compatible with C++ AMP" << std::endl;
		list_accelerators();
	}
} // query_AMP_support

/*
This function transforms uniformly distributed variables to normally distributed variables
using the Cartesian form Box Muller transform. Box Muller is inferior in speed to Ziggurat
algorithm but simpler to implemnt. That's why I've chosen Box Muller over Ziggurat algorithm.
Snippet is adapted from a Microsoft sample. See https://goo.gl/cU6b1X for details.
*/
void box_muller_transform(float& u1, float& u2) restrict(amp)
{
	float r = fast_math::sqrt(-2.0f * fast_math::log(u1));
	float phi = 2.0f * 3.14159265358979f * u2;
	u1 = r * fast_math::cos(phi);
	u2 = r * fast_math::sin(phi);
} // box_muller_transform

/* This function calculates the value at risk from a given set of endValues by sorting a given list
and extracting a value at rank x. Functionality is identifical to nth_rank in stl. amp_stl_algorithms
contains a function called nth_element, but it is not implemented as of today. My own implementation
uses radix select implementation with a complexity of O(n), which is an adoption of radix sort from
the amp algorithms library. See https://archive.codeplex.com/?p=ampalgorithms for details.
*/
void calculate_value_at_risk(array<float>& endValues, unsigned const int rank) restrict(amp) {
	// todo: implementation using radix select.
} // calculate_value_at_risk

/* This function calculates random paths using geometric brownian motion (GBM) for a given holding period. For
details on geometric brownian motion see: https://goo.gl/lrCeLJ.
*/
void generate_random_paths(const unsigned int seed, const int size, const float initialValue, const float expectedReturn, const float volatility, const float tradingDays, std::vector<float>& endValues) {
	const int TS = 1024;
	const unsigned int RANK = 1;
	
	// validate that given input is optimal
	static_assert((HOLDING_PERIOD % 2 == 0), "The holding period must be a multiple of two.");
	static_assert((TS % 2 == 0 && TS >= 2), "Tilesize must be a multiple of two.");
	
	// todo: find out what extent is best for tinymyt_collection, large numbers lead to crash of program
	extent<RANK> tinyE(4096);
	tinymt_collection<RANK> randCollection(tinyE, seed);

	extent<RANK> e(size);
	array_view<float> endvaluesAv(e, endValues);

	// do not copy data to accelerator
	endvaluesAv.discard_data();

	// start clock for GPU version after array allocation
	the_amp_clock::time_point start = the_amp_clock::now();

	// wrap parallel_for_each in try catch to provide feedback on runtime exceptions
	try {
		parallel_for_each(endvaluesAv.extent.tile<TS>(), [=](tiled_index<TS>t_idx) restrict(amp) {
			index<1> idx = t_idx.global;
			auto t = randCollection[idx];
			float s(0.0f);
			float prevS(initialValue);

			// see https://goo.gl/Rb394n for rationelle behind modifying drift and volatility.
			// scale drift to timestep
			const float dailyDrift = expectedReturn / tradingDays;
			// scale volatility to timestep. Volatility scales with square root of time.
			const float dailyVolatility = volatility / fast_math::sqrt(tradingDays);
			// extract volatility from daily drift
			const float meanDrift = dailyDrift - 0.5f * dailyVolatility * dailyVolatility;

			// generate path for entire holding period, write endprices back to vector
			for (auto day(1); day <= HOLDING_PERIOD / 2; day++) {
				// generate two random numbers and convert to normally distributed numbers
				auto z0 = t.next_single();
				auto z1 = t.next_single();
				box_muller_transform(z0, z1);

				// Using loop unrolling for performance optimizatation, limit minimum price to 0
				float ds = meanDrift + dailyVolatility * z0;
				s = fast_math::fmax(prevS * fast_math::expf(ds), 0.0f);
				prevS = s;

				ds = meanDrift + dailyVolatility * z1;
				s = fast_math::fmax(prevS * fast_math::expf(ds), 0.0f);
				prevS = s;

			}
			endvaluesAv[idx] = s;
		});
		endvaluesAv.synchronize();
		// Stop timing
		the_amp_clock::time_point end = the_amp_clock::now();
		// Compute the difference between the two times in milliseconds
		auto time_taken = duration_cast<milliseconds>(end - start).count();
		std::cout << "Calculating random paths took: " << time_taken << " ms." << std::endl;
	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}
} // generate_random_paths

/*
Helper function to avoid lazy initialization and just in time overhead (JIT) on first execution.
For details see: https://goo.gl/DPZuGU
*/
void warm_up() {
	//fill vector
	std::vector<float> v1(1024, 0.0f);
	// run kernel with minimal dataset
	generate_random_paths(0, 1024, 0, 0, 0, 0, v1);
} // warm_up

int main(int argc, char *argv[])
{
	// Check AMP support
	query_AMP_support();
	// run kernel once on small dataset to supress effects of lazy init and jit.
	warm_up();

	// create vector holding all paths
	std::vector<float> pathVector(NUMBER_OF_PATHS, 0);

	const unsigned int seed(7859);
	generate_random_paths(seed, NUMBER_OF_PATHS, INITIAL_VALUE, EXPECTED_RETURN, VOLATILITY, TRADING_DAYS, pathVector);

	// print the first view results for validation
	for (auto i(0); i < 100; i++) {
		std::cout << std::fixed << std::setprecision(5) << pathVector[i] << std::endl;
	}
	return 0;
} // main