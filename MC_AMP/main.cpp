
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


// Need to access the concurrency libraries 
using namespace concurrency;

// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;

// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_serial_clock;
typedef std::chrono::steady_clock the_amp_clock;

#define TS 32

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
void generate_random_paths(const unsigned seed, const int size, const float initialValue, const float expectedReturn, const float volatility, const int tradingDays, const int holdingPeriod, concurrency::array_view<float>& endvaluesAv) {

	// validate that given input is optimal
	assert(holdingPeriod % 2 == 0);

	/* todo: find out what extent is best for tinymyt_collection, large numbers lead to crash of program probably due
	/ to memory limitations. If solved change auto t to use global idx. Small tiny collection delivers falsy results
	meaning that a higher number of paths won't deliver a higher accurancy.*/
	const extent<1> tinyE(65'536);
	const tinymt_collection<1> randCollection(tinyE, seed);

	// start clock for GPU version after array allocation
	the_amp_clock::time_point start = the_amp_clock::now();

	// wrap parallel_for_each in try catch to provide feedback on runtime exceptions
	try {
		parallel_for_each(endvaluesAv.extent, [=](index<1>idx) restrict(amp) {

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
			endvaluesAv[idx] = s;
		});
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

/* This function returns the smallest element of an array_view calculated using an
map reduce approach. The major part is calculated on the gpu. While the summing
of the tiles is done on the gpu. In case of an error 0 is returned.
*/
float min_element(concurrency::array_view<float, 1>& avSrc, int elementCount) {

	// todo: support array view with other tile_size as well

	// tile_size and tile_count are not matching element_count
	assert(elementCount % TS == 0);
	// element_count is not valid.
	assert(elementCount < 0 && elementCount <= INT_MAX);

	// Using arrays as temporary memory. Array holds at least one lement
	array<float> arr((elementCount / TS) ? (elementCount / TS) : 1);
	array_view<float> av_dst(arr);
	// do not copy data to gpu
	av_dst.discard_data();

	// start clock for GPU version after array allocation
	the_amp_clock::time_point start = the_amp_clock::now();
	try
	{
		// Reduce using parallel_for_each as long as the sequence length
		// is evenly divisable to the number of threads in the tile.
		while ((elementCount % TS) == 0)
		{
			parallel_for_each(extent<1>(elementCount).tile<TS>(), [=](tiled_index<TS> tidx) restrict(amp)
			{
				// Use tile_static as a scratchpad memory.
				tile_static float tile_data[TS];

				unsigned local_idx = tidx.local[0];
				tile_data[local_idx] = avSrc[tidx.global];
				tidx.barrier.wait();

				for (unsigned s = TS / 2; s > 0; s /= 2) {
					if (local_idx < s) {
						//tile_data[local_idx] += tile_data[local_idx + s];
						tile_data[local_idx] = fast_math::fmin(tile_data[local_idx], tile_data[local_idx + s]);
					}
					tidx.barrier.wait();
				}
				// Store the tile result in the global memory.
				if (local_idx == 0)
				{
					av_dst[tidx.tile] = tile_data[0];
				}
			});
			// Update the sequence length, swap source with destination.
			elementCount /= TS;
			std::swap(avSrc, av_dst);
		}
		av_dst.discard_data();
		// Perform any remaining reduction on the CPU.
		std::vector<float> result(elementCount);

		// copy only part of array_view back to host, which contains the minimal elements (for performance reasons)
		copy(avSrc.section(0, elementCount), result.begin());

		// Stop timing
		the_amp_clock::time_point end = the_amp_clock::now();

		// Compute the difference between the two times in milliseconds
		auto time_taken = duration_cast<milliseconds>(end - start).count();
		std::cout << "Obtaining the smallest result took: " << time_taken << " ms." << std::endl;
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
void calculate_value_at_risk(const int numberOfPaths, const float initialValue, const float expectedReturn, const float volatility, const int tradingDays, const int holdingPeriod, const int seed = 7859) {

	// initialize array view, do not copy data to accelerator
	extent<1> e(numberOfPaths);
	array_view<float> endvaluesAv(e);
	endvaluesAv.discard_data();

	// first kernel: generate random paths
	generate_random_paths(seed, numberOfPaths, initialValue, expectedReturn, volatility, tradingDays, holdingPeriod, endvaluesAv);

	// second kernel: rearrange elements to obtain element at rank
	float minResult = min_element(endvaluesAv, numberOfPaths);

	// print value at risk
	std::cout << "Value at risk at " << holdingPeriod << " days with " << "100 % confidence: "
		<< minResult - initialValue << " GPB (with - being risk and + being chance)" << std::endl;
} // calculate_value_at_risk


  /*
  Helper function to avoid lazy initialization and just in time overhead (JIT) on first execution.
  For details see: https://goo.gl/DPZuGU
  */
void warm_up() {
	// run kernel with minimal dataset
	calculate_value_at_risk(1024, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
} // warm_up

int main(int argc, char *argv[])
{
	// Check AMP support
	query_AMP_support();
	// run kernel once on small dataset to supress effects of lazy init and jit.
	warm_up();
	for (auto i(1'024); i <= 4'096; i *= 2) {
		// create vector holding all paths
		calculate_value_at_risk(i, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
	}
	return 0;
} // main

