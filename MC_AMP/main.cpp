#define NOMINMAX

#include <fstream>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <string>
#include <amp_tinymt_rng.h>
#include <amp_math.h>
#include <amp_algorithms.h>
#include <fstream>
#include <cvmarkersobj.h>
#include <tclap/CmdLine.h>

// Need to access the concurrency libraries 
using namespace concurrency;
using namespace diagnostic;
// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;

// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_serial_clock;
typedef std::chrono::steady_clock the_amp_clock;

std::ofstream file;
// Define variables needed for value at risk calculation

marker_series markers;

void report_accelerator(const accelerator a)
{
	const std::wstring bs[2] = { L"false", L"true" };
	std::wcout << ": " << a.description << " "
		<< std::endl << "       device_path                       = " << a.device_path
		<< std::endl << "       dedicated_memory                  = " << std::setprecision(4) << float(a.dedicated_memory) / (
			1024.0f * 1024.0f) << " Mb"
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
	for (const auto& a : accls)
	{
		report_accelerator(a);
	}

	const accelerator acc = accelerator(accelerator::default_accelerator);
	std::wcout << " default acc = " << acc.description << std::endl;
	// todo: replace with assert?
	if (acc == accelerator(accelerator::direct3d_ref))
		std::wcout << "Running on very slow emulator! Only use this accelerator for debugging." << std::endl;
} // list_accelerators

// query if AMP accelerator exists on hardware
void query_amp_support()
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
	const float r = fast_math::sqrt(-2.0f * fast_math::log(u1));
	const float phi = 2.0f * 3.14159265358979f * u2;
	u1 = r * fast_math::cos(phi);
	u2 = r * fast_math::sin(phi);
} // box_muller_transform

/* This function calculates random paths using geometric brownian motion (GBM) for a given holding period. For
details on geometric brownian motion see: https://goo.gl/lrCeLJ.
*/
void generate_random_paths(const unsigned seed, const float initial_value, const float expected_return,
	const float volatility, const int trading_days, const int holding_period,
	array<float>& endvalues)
{
	// validate that given input is optimal
	assert(holdingPeriod % 2 == 0);

	/* todo: find out what extent is best for tinymyt_collection, large numbers lead to crash of program probably due
	/ to memory limitations. If solved change auto t to use global idx. Small tiny collection delivers falsy results
	meaning that a higher number of paths won't deliver a higher accurancy.*/

	const extent<1> tiny_e(65'536);
	const tinymt_collection<1> rand_collection(tiny_e, seed);

	// flag for concurrency visualizer
	markers.write_flag(normal_importance, L"generate random");

	// wrap parallel_for_each in try catch to provide feedback on runtime exceptions
	try
	{
		parallel_for_each(endvalues.extent, [=, &endvalues](index<1> idx) restrict(amp)
		{
			float s(0.0f);
			float prev_s(initial_value);
			auto t = rand_collection[idx % 16'384];

			// see https://goo.gl/Rb394n for rationelle behind modifying drift and volatility.
			// scale drift to timestep
			const float daily_drift = expected_return / trading_days;
			// scale volatility to timestep. Volatility scales with square root of time.
			// Use rsqrt for performance reasons (See Chapter 7 AMP-Book)
			const float daily_volatility = volatility * fast_math::rsqrtf(static_cast<float>(trading_days));
			// extract volatility from daily drift
			const float mean_drift = daily_drift - 0.5f * daily_volatility * daily_volatility;
			// generate path for entire holding period, write endprices back to vector
			for (auto day(1); day <= holding_period / 2; day++)
			{
				// generate two random numbers and convert to normally distributed numbers
				auto z0 = t.next_single();
				auto z1 = t.next_single();
				box_muller_transform(z0, z1);

				// Using loop unrolling for performance optimizatation, limit minimum price to 0
				float ds = mean_drift + daily_volatility * z0;
				s = fast_math::fmax(prev_s * fast_math::expf(ds), 0.0f);
				prev_s = s;

				ds = mean_drift + daily_volatility * z1;
				s = fast_math::fmax(prev_s * fast_math::expf(ds), 0.0f);
				prev_s = s;
			}
			endvalues[idx] = s;
		});
	}
	catch (const runtime_exception& ex)
	{
		MessageBoxA(nullptr, ex.what(), "Error", MB_ICONERROR);
	}
} // generate_random_paths


/* This function returns the smallest element of an array_view calculated using an
map reduce approach. The major part is calculated on the gpu. While the summing
of the tiles is done on the gpu. In case of an error 0 is returned.
Tile size needs to be known at compile time. That's why I am using template arguments
here.
*/
template <const int TileSize>
float min_element(array<float, 1>& src, int element_count)
{
	// check for max tile size
	assert(tile_size >= 2 && tile_size <= 1'024);
	// tile_size and tile_count are not matching element_count
	assert(elementCount % TS == 0);
	// element_count is not valid.
	assert(elementCount < 0 && elementCount <= INT_MAX);
	// check if number of tiles is <= 65k, which is the max in AMP
	assert(elementCount / TS < 65'536);

	// Using arrays as temporary memory. Array holds at least one lement
	array<float, 1> dst(element_count / TileSize ? element_count / TileSize : 1);

	markers.write_flag(normal_importance, L"reduce");

	try
	{
		// Reduce using parallel_for_each as long as the sequence length
		// is evenly divisable to the number of threads in the tile.
		while (element_count % TileSize == 0)
		{
			parallel_for_each(extent<1>(element_count).tile<TileSize>(),
				[=, &src, &dst](tiled_index<TileSize> tidx) restrict(amp)
			{
				// Use tile_static as a scratchpad memory.
				tile_static float tile_data[TileSize];

				unsigned local_idx = tidx.local[0];
				tile_data[local_idx] = src[tidx.global];
				tidx.barrier.wait();

				for (unsigned s = TileSize / 2; s > 0; s /= 2)
				{
					if (local_idx < s)
					{
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
			element_count /= TileSize;
			std::swap(src, dst);
		}
		// Perform any remaining reduction on the CPU.
		std::vector<float> result(element_count);

		// copy only part of array_view back to host, which contains the minimal elements (for performance reasons)
		copy(src.section(0, element_count), result.begin());

		// reduce all remaining tiles on the cpu
		const auto idx = std::min_element(result.begin(), result.end());
		return result.at(idx - result.begin());
	}
	catch (const runtime_exception& ex)
	{
		MessageBoxA(nullptr, ex.what(), "Error", MB_ICONERROR);
	}
	return 0;
} // min_element

/* This function calculates the value at risk at a confidence level of 100 % by calling the generate_random_paths function and by extracting
the endvalue at rank 0. The functionality is similar to stl::min_element()*/
template <const int TileSize>
void calculate_value_at_risk(std::vector<float>& host_end_values, const float initial_value,
	const float expected_return,
	const float volatility, const int trading_days, const int holding_period,
	const int seed)
{
	// time taken to initialize, no copying of data, entire implementation uses array instead of array_view to be able to measure copying times as well (cmp. p. 131 AMP book) 
	const auto start_initialize = the_amp_clock::now();
	array<float> gpu_end_values(host_end_values.size());
	gpu_end_values.accelerator_view.wait();
	const auto end_initialize = the_amp_clock::now();
	const auto elapsed_time_initialize = duration_cast<milliseconds>(end_initialize - start_initialize).count();
	std::cout << std::setw(35) << std::left << "Initialize time: " << elapsed_time_initialize << std::endl;

	// first kernel: generate random paths
	generate_random_paths(seed, initial_value, expected_return, volatility, trading_days, holding_period,
		gpu_end_values);
	gpu_end_values.accelerator_view.wait();
	auto const end_kernel_one = the_amp_clock::now();
	const auto elapsed_time_kernel_one = duration_cast<milliseconds>(end_kernel_one - end_initialize).count();
	std::cout << std::setw(35) << std::left << "Kernel one time: " << elapsed_time_kernel_one << std::endl;

	// write endvalues back to host for further investigation
	copy(gpu_end_values, host_end_values.begin());
	gpu_end_values.accelerator_view.wait();
	auto const end_copy = the_amp_clock::now();
	auto const elapsed_time_copying = duration_cast<milliseconds>(end_copy - end_kernel_one).count();
	std::cout << std::setw(35) << std::left << "copying time: " << elapsed_time_copying << std::endl;

	// second kernel: rearrange elements to obtain element at rank 0
	auto min_result = min_element<TileSize>(gpu_end_values, host_end_values.size());
	gpu_end_values.accelerator_view.wait();
	const auto end_kernel_two = the_amp_clock::now();
	const auto elapsed_time_kernel_two = duration_cast<milliseconds>(end_kernel_two - end_copy).count();
	std::cout << std::setw(35) << std::left << "Kernel two time: " << elapsed_time_kernel_two << std::endl;

	// total elapsed time. It can slightly differ from the individual times due to casting
	const auto elapsed_time_total = duration_cast<milliseconds>(end_kernel_two - start_initialize).count();

	// write time to file
	//file << elapsed_time_kernel_two << ",";
	//file << elapsedTimeInitialize << "," << elapsedTimeKernelOne << "," << elapsedTimeCopying << "," << elapsedTimeKernelTwo;

	std::cout << std::setw(35) << std::left << "Total time: " << elapsed_time_total << std::endl << std::endl;

	// print value at risk
	std::cout << "Value at risk at " << holding_period << " days with " << "100 % confidence: "
		<< min_result - initial_value << " GPB (with - being risk and + being chance)" << std::endl;
} // calculate_value_at_risk

/*
Helper function to avoid lazy initialization and just in time overhead (JIT) on first execution.
For details see: https://goo.gl/DPZuGU */
void warm_up()
{
	std::vector<float> paths(1024, 0);
	// run kernel with minimal dataset
	calculate_value_at_risk<4>(paths, 10.0f, 0.05f, 0.04f, 300, 300, 7859);
	std::cout <<
		"------------------------------------- valid results starting from here -------------------------------------" << std
		::endl;
} // warm_up

/* This is wrapper function around calculate_value_at_risk. It lets the tile_size dynamically
by using template parameters. The tilesize must be known at compile time. An approach similar
to this is suggested in the AMP book.
*/
void run(const unsigned& tile_size, std::vector<float>& paths, const float initial_value = 10.0f,
	const float expected_return = 0.05f, const float volatility = 0.04f, const int trading_days = 300,
	const int holding_period = 300, const int seed = 7'859)
{
	switch (tile_size)
	{
	case 2:
		calculate_value_at_risk<2>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 4:
		calculate_value_at_risk<4>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 8:
		calculate_value_at_risk<8>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 16:
		calculate_value_at_risk<16>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 32:
		calculate_value_at_risk<32>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 64:
		calculate_value_at_risk<64>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 128:
		calculate_value_at_risk<128>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 256:
		calculate_value_at_risk<256>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 512:
		calculate_value_at_risk<512>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	case 1024:
		calculate_value_at_risk<1024>(paths, initial_value, expected_return, volatility, trading_days, holding_period, seed);
		break;
	default:
		assert(false);
	}
} // run

int main(int argc, char* argv[])
{
	/*
	try
	{
		TCLAP::CmdLine cmd("AMPMC", ' ', "1");

		// Define the arguments
		TCLAP::ValueArg<float> initial_value("i", "initial_value", "Initial value of the investment.", false, 10.0f, "float");
		TCLAP::ValueArg<float> annual_return("r", "annual_return", "Annual return of the investment", false, 0.05f, "float");
		TCLAP::ValueArg<float> annual_volatility("v", "annual_volatility", "Annual volalitility of the investment", false,
												 0.05f, "float");
		TCLAP::ValueArg<int> trading_days("t", "trading_days", "Annual trading days", false, 300, "int");
		TCLAP::ValueArg<int> holding_period("d", "duration", "Duration of the investment", false, 300, "int");
		TCLAP::ValueArg<int> seed("s", "seed", "Seed for random number generator", false, 7'859, "int");
		TCLAP::ValueArg<int> paths("p", "paths", "Number of paths", false, 1'024, "int");
		TCLAP::ValueArg<int> tile_size("x", "tile_size", "Size of tiles", false, 16, "int");
		cmd.add(initial_value);
		cmd.add(annual_return);
		cmd.add(annual_volatility);
		cmd.add(trading_days);
		cmd.add(holding_period);
		cmd.add(seed);
		cmd.add(paths);
		cmd.add(tile_size);
		// Parse arguments
		cmd.parse(argc, argv);

		// Check AMP support
		query_amp_support();

		// run kernel once on small dataset to supress effects of lazy init and jit.
		warm_up();

		// run kernel as set by specified arguments
		std::vector<float> path_vector(paths.getValue());
		run(tile_size.getValue(), path_vector, initial_value.getValue(), annual_return.getValue(),
			annual_volatility.getValue(), trading_days.getValue(), holding_period.getValue(), seed.getValue());
	}
	catch (TCLAP::ArgException& e)
	{
		std::cout << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}
	*/
	// Check AMP support
	//query_AMP_support();
	// run kernel once on small dataset to supress effects of lazy init and jit.
	//warm_up();
	/*
	// start multi comparsion
	file.open("measures.csv", std::ios::out);
	// prepare header
	file << "v ps : > ts,";
	for (auto ts(16); ts <= 1'024; ts *= 2)
		file << ts << ",";
	file << std::endl;
	*/

	/* prepare body, dimensions is due to the limitations on tile size and tile count of c++ AMP.
	See https://bit.ly/2qgCeTB for details.*/
	/*
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
	*/
	// test for concurrency visualizer
	query_amp_support();
	warm_up();
	int ps(524'288), ts(128);
	std::vector<float>paths(ps);
	run(ts, paths);

	return 0;
} // main
