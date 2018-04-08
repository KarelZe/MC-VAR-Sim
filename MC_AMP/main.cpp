
#define NOMINMAX

#include <fstream>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <time.h>
#include <string>
#include <amp_tinymt_rng.h>
#include <amp_math.h>
#include <amp_algorithms.h>
#include <fstream>

// Need to access the concurrency libraries 
using namespace concurrency;

// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;

// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_serial_clock;
typedef std::chrono::steady_clock the_amp_clock;

std::ofstream file;
// Define variables needed for value at risk calculation

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
	for (unsigned i(0); i < accls.size(); i++)
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

  /* This function calculates random paths using geometric brownian motion (GBM) for a given holding period. For
  details on geometric brownian motion see: https://goo.gl/lrCeLJ.
  */
void generate_random_paths(const unsigned seed, const int size, const float initialValue, const float expectedReturn, const float volatility, const int tradingDays, const int holdingPeriod, concurrency::array<float>& endvalues) {

	// validate that given input is optimal
	assert(holdingPeriod % 2 == 0);

	/* todo: find out what extent is best for tinymyt_collection, large numbers lead to crash of program probably due
	/ to memory limitations. If solved change auto t to use global idx. Small tiny collection delivers falsy results
	meaning that a higher number of paths won't deliver a higher accurancy.*/
	const extent<1> tinyE(65'536);
	const tinymt_collection<1> randCollection(tinyE, seed);

	// start clock for GPU version after array allocation
	//the_amp_clock::time_point start = the_amp_clock::now();

	// wrap parallel_for_each in try catch to provide feedback on runtime exceptions
	try {
		parallel_for_each(endvalues.extent, [=, &endvalues](index<1>idx) restrict(amp) {

			float s(0.0f);
			float prevS(initialValue);
			auto t = randCollection[idx % 16'384];

			// see https://goo.gl/Rb394n for rationelle behind modifying drift and volatility.
			// scale drift to timestep
			const float dailyDrift = expectedReturn / tradingDays;
			// scale volatility to timestep. Volatility scales with square root of time.
			// Use rsqrt for performance reasons (See Chapter 7 AMP-Book)
			const float dailyVolatility = volatility * fast_math::rsqrtf(static_cast<float>(tradingDays));
			// extract volatility from daily drift
			const float meanDrift = dailyDrift - 0.5f * dailyVolatility * dailyVolatility;
			// generate path for entire holding period, write endprices back to vector
			for (auto day(1); day <= holdingPeriod / 2; day++) {

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
			endvalues[idx] = s;
		});
	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}
} // generate_random_paths

/* Function tries reading from array_view. If index is part of array_view it's value is returned,
otherwise the greatest float*/
float padded_read(const array_view<float, 1>& src, const index<1>& idx) restrict(cpu, amp)
{
	return src.extent.contains(idx) ? src[idx] : FLT_MAX;
} // padded_read
/* Function tries to write to array_view, if index is part of source array_view.
*/
void padded_write(const array_view<float>& src, const index<1>& idx, const float& val) restrict(cpu, amp)
{
	if (src.extent.contains(idx))
		src[idx] = val;
} // padded_write

/* This function returns the smallest element of an array_view calculated using an
map reduce approach. The major part is calculated on the gpu. While the summing
of the tiles is done on the gpu. In case of an error 0 is returned.
Tile size needs to be known at compile time. That's why I am using template arguments
here.
*/
template<const int tile_size>
float min_element(concurrency::array<float, 1>& src, int elementCount) {

	// check for max tile size
	assert(tile_size >= 2 && tile_size <= 1'024)
		// tile_size and tile_count are not matching element_count
		assert(elementCount % TS == 0);
	// element_count is not valid.
	assert(elementCount < 0 && elementCount <= INT_MAX);
	// check if number of tiles is <= 65k, which is the max in AMP
	assert(elementCount / TS < 65'536);

	// Using arrays as temporary memory. Array holds at least one lement
	array<float> dst((elementCount / tile_size) ? (elementCount / tile_size) : 1);

	try
	{
		// Reduce using parallel_for_each as long as the sequence length
		// is evenly divisable to the number of threads in the tile.
		while ((elementCount % tile_size) == 0)
		{
			parallel_for_each(extent<1>(elementCount).tile<tile_size>(), [=, &src, &dst](tiled_index<tile_size> tidx) restrict(amp)
			{
				// Use tile_static as a scratchpad memory.
				tile_static float tile_data[tile_size];

				unsigned local_idx = tidx.local[0];
				tile_data[local_idx] = src[tidx.global];
				tidx.barrier.wait();

				for (unsigned s = tile_size / 2; s > 0; s /= 2) {
					if (local_idx < s) {
						tile_data[local_idx] = fast_math::fmin(tile_data[local_idx], tile_data[local_idx + s]);
					}
					tidx.barrier.wait();
				}
				// Store the tile result in the global memory.
				if (local_idx == 0)
				{
					dst[tidx.tile] = tile_data[0];
				}
			});
			// Update the sequence length, swap source with destination.
			elementCount /= tile_size;
			std::swap(src, dst);
		}
		// Perform any remaining reduction on the CPU.
		std::vector<float> result(elementCount);

		// copy only part of array_view back to host, which contains the minimal elements (for performance reasons)
		copy(src.section(0, elementCount), result.begin());

		// reduce all remaining tiles on the cpu
		auto idx = std::min_element(result.begin(), result.end());
		return result.at(idx - result.begin());
	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}
	return 0;
} // min_element

  /* This function calculates the value at risk at a confidence level of 100 % by calling the generate_random_paths function and by extracting
  the endvalue at rank 0. The functionality is similar to stl::min_element()*/
template<const int tile_size>
void calculate_value_at_risk(std::vector<float>hostEndValues, const float initialValue, const float expectedReturn, const float volatility, const int tradingDays, const int holdingPeriod, const int seed = 7859) {

	// time taken to initialize, no copying of data, entire implementation uses array instead of array_view to be able to measure copying times as well (cmp. p. 131 AMP book) 
	the_amp_clock::time_point startInitialize = the_amp_clock::now();
	array<float> gpuEndValues(hostEndValues.size());
	gpuEndValues.accelerator_view.wait();
	the_amp_clock::time_point endInitialize = the_amp_clock::now();
	auto elapsedTimeInitialize = duration_cast<milliseconds>(endInitialize - startInitialize).count();
	std::cout << std::setw(35) << std::left << "Initialize time: " << elapsedTimeInitialize << std::endl;

	// first kernel: generate random paths
	generate_random_paths(seed, hostEndValues.size(), initialValue, expectedReturn, volatility, tradingDays, holdingPeriod, gpuEndValues);
	gpuEndValues.accelerator_view.wait();
	the_amp_clock::time_point endKernelOne = the_amp_clock::now();
	auto elapsedTimeKernelOne = duration_cast<milliseconds>(endKernelOne - endInitialize).count();
	std::cout << std::setw(35) << std::left << "Kernel one time: " << elapsedTimeKernelOne << std::endl;

	// write endvalues back to host for further investigation
	copy(gpuEndValues, hostEndValues.begin());
	gpuEndValues.accelerator_view.wait();
	the_amp_clock::time_point endCopy = the_amp_clock::now();
	auto elapsedTimeCopying = duration_cast<milliseconds>(endCopy - endKernelOne).count();
	std::cout << std::setw(35) << std::left << "copying time: " << elapsedTimeCopying << std::endl;

	// second kernel: rearrange elements to obtain element at rank 0
	float minResult = min_element<tile_size>(gpuEndValues, hostEndValues.size());
	gpuEndValues.accelerator_view.wait();
	the_amp_clock::time_point endKernelTwo = the_amp_clock::now();
	auto elapsedTimeKernelTwo = duration_cast<milliseconds>(endKernelTwo - endCopy).count();
	std::cout << std::setw(35) << std::left << "Kernel two time: " << elapsedTimeKernelTwo << std::endl;

	// total elapsed time. It can slightly differ from the individual times due to casting
	auto elapsedTimeTotal = duration_cast<milliseconds>(endKernelTwo - startInitialize).count();

	// file << elapsedTimeTotal << ",";
	// file << elapsedTimeInitialize << "," << elapsedTimeKernelOne << "," << elapsedTimeCopying << "," << elapsedTimeKernelTwo;

	// write time to file
	std::cout << std::setw(35) << std::left << "Total time: " << elapsedTimeTotal << std::endl << std::endl;

	// print value at risk
	//std::cout << "Value at risk at " << holdingPeriod << " days with " << "100 % confidence: "
	//	<< minResult - initialValue << " GPB (with - being risk and + being chance)" << std::endl;
} // calculate_value_at_risk

/*
Helper function to avoid lazy initialization and just in time overhead (JIT) on first execution.
For details see: https://goo.gl/DPZuGU */
void warm_up() {
	std::vector<float>paths(1024, 0);
	// run kernel with minimal dataset
	calculate_value_at_risk<4>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
	std::cout << "------------------------------------- valid results starting from here -------------------------------------" << std::endl;
} // warm_up

/* This is wrapper function around calculate_value_at_risk. It lets the tile_size dynamically
by using template parameters. The tilesize must be known at compile time. An approach similar
to this is suggested in the AMP book.
*/
void run(const unsigned &tile_size, std::vector<float> &paths) {
	switch (tile_size) {
	case 2:
		calculate_value_at_risk<2>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 4:
		calculate_value_at_risk<4>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 8:
		calculate_value_at_risk<8>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 16:
		calculate_value_at_risk<16>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 32:
		calculate_value_at_risk<32>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 64:
		calculate_value_at_risk<64>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 128:
		calculate_value_at_risk<128>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 256:
		calculate_value_at_risk<256>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 512:
		calculate_value_at_risk<512>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	case 1024:
		calculate_value_at_risk<1024>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
		break;
	default:
		assert(false);
	}
} // run

int main(int argc, char *argv[])
{
	// Check AMP support
	query_AMP_support();
	// run kernel once on small dataset to supress effects of lazy init and jit.
	warm_up();

	// start multi comparsion
	file.open("measures.csv", std::ios::out);
	// prepare header
	file << "v ps : > ts,";
	for (auto ts(16); ts <= 1'024; ts *= 2)
		file << ts << ",";
	file << std::endl;
	// file << "Initialize time,kernel one time, copying time, kernel two time"<<std::endl;

	/* prepare body, dimensions is due to the limitations on tile size and tile count of c++ AMP.
	See https://bit.ly/2qgCeTB for details.*/
	for (auto ps(1024); ps <= 524'288; ps *= 2) {
		// initialize vector of problemsize, will contain endprices later.
		std::vector<float>paths(ps);
		// write problem size to first column
		file << ps << ",";
		for (auto ts(16); ts <= 1'024; ts *= 2) {
			run(ts, paths);
		}
		file << std::endl;
	}
	// close file stream
	file.close();
	return 0;
} // main




