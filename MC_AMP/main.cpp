#define NOMINMAX
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <amp.h>
#include <amp_tinymt_rng.h>
#include <amp_math.h>
#include <cvmarkersobj.h>
#include <tclap/CmdLine.h>
#include <PercentageConstraint.h>
#include <IntegerConstraint.h>
#include <EvenIntegerConstraint.h>
#include <PathConstraint.h>
#include <TileConstraint.h>

using namespace concurrency;
using namespace diagnostic;

using std::chrono::duration_cast;
using std::chrono::microseconds;

typedef std::chrono::steady_clock the_serial_clock;
typedef std::chrono::steady_clock the_amp_clock;


// std::ofstream file;
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
	for (const auto& accl : accls)
	{
		report_accelerator(accl);
	}

	const accelerator acc = accelerator(accelerator::default_accelerator);
	std::wcout << " default acc = " << acc.description << std::endl << std::endl;
	if (acc == accelerator(accelerator::direct3d_ref))
		std::wcout << "Running on very slow emulator! Only use this accelerator for debugging." << std::endl;
} // list_accelerators

// query if AMP accelerator exists on hardware and print all available accelerators
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
using the Cartesian form of Box Muller transform. Box Muller is inferior in speed to Ziggurat
algorithm but simpler to implement. That's why I've chosen Box Muller over Ziggurat algorithm.
Snippet is adapted from a Microsoft sample. See https://goo.gl/cU6b1X for details.
*/
inline void box_muller_transform(float& u1, float& u2) restrict(amp)
{
	// limit minimum uniform random to (0,1], as log(0) = n. d.
	u1 = fast_math::fmax(u1, 0.00000000000001f);
	const float r = fast_math::sqrt(-2.0f * fast_math::log(u1));
	const float phi = 6.28318530717958f * u2;
	u1 = r * fast_math::cos(phi);
	u2 = r * fast_math::sin(phi);
} // box_muller_transform

/* This function calculates random paths using geometric brownian motion (GBM) for a given holding period. For
details on geometric brownian motion see: https://goo.gl/lrCeLJ.
*/
void generate_random_paths(const float initial_value, const float expected_return,
	const float volatility, const int trading_days, const int holding_period,
	array<float>& endvalues)
{
	// validate that given input is optimal
	assert(holding_period % 2 == 0 && holding_period >= 2);
	assert(initial_value > 0);
	assert(trading_days > 0);
	assert(volatility > 0 && volatility <= 1);
	assert(expected_return > 0 && expected_return < 1);
	/* tinymt_collection is a wrapper around concurrency::array. Its maximum extent is at 65'536 due to the way how the rng
	 * is implemented. I am copying the same random generator for all threads but seeding it differently with the thread id.
	 */
	const extent<1> tinymt_extent(1);
	const tinymt_collection<1> tinymt_collection(tinymt_extent);

	// flag for concurrency visualizer
	markers.write_flag(normal_importance, L"generate random");

	// wrap parallel_for_each in try catch to provide feedback on runtime exceptions
	try
	{
		parallel_for_each(endvalues.extent, [=, &endvalues](index<1> idx) restrict(amp)
		{
			float s(0.0f);
			float prev_s(initial_value);
			// copy rng instance from host
			tinymt t = tinymt_collection[index<1>(0)];
			// seed rng with index
			t.initialize(idx[0]);
			// implementation follows a Geometric brownian motion model. See https://bit.ly/2HLeqPS for considerations behind it.
			// scale drift to timestep
			const float daily_drift = expected_return / trading_days;
			// scale volatility to timestep. Volatility scales with square root of time.
			// Use rsqrtf for performance reasons (See p. 163 AMP-Book)
			const float daily_volatility = volatility * fast_math::rsqrtf(static_cast<float>(trading_days));
			// extract volatility from daily drift
			const float mean_drift = daily_drift - 0.5f * daily_volatility * daily_volatility;
			// generate path for entire holding period, write endprices back to array
			for (auto day(1); day <= holding_period / 2; day++)
			{
				// generate two random numbers between 0-1 and convert to normally distributed numbers
				float z0 = t.next_single();
				float z1 = t.next_single();
				box_muller_transform(z0, z1);

				// Using loop unrolling for performance optimizatation, limit minimum price to 0
				float ds = mean_drift + daily_volatility * z0;
				s = fast_math::fmaxf(prev_s * fast_math::expf(ds), 0.0f);
				prev_s = s;

				ds = mean_drift + daily_volatility * z1;
				s = fast_math::fmaxf(prev_s * fast_math::expf(ds), 0.0f);
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
map reduce approach. The major part is calculated on the gpu. In case of an error 0 is returned.
Tile size needs to be known at compile time. That's why I am using template arguments
here. The implementation is inspired by implementation discussed in class (See Falconer, p. 25),
but already performs the first reduction while saving to tile_static leading to an performance
improvement of ca. 50 %. Implementation inspired by Reduction case study in AMP book.
*/
template <const int TileSize>
float min_element(array<float, 1>& src)
{
	// get total element_count
	int element_count = src.extent[0];
	// Using arrays as temporary memory. Array holds at least one element
	array<float, 1> dst(element_count / TileSize ? element_count / TileSize : 1);

	// check for max tile size
	assert(TileSize >= 2 && TileSize <= 1'024);
	// tile_size and tile_count are not matching element_count
	assert((element_count % TileSize) == 0);
	// element_count is not valid.
	assert(element_count > 0 && element_count <= INT_MAX);
	// check if number of tiles is <= 65k, which is the max in AMP
	assert((element_count / TileSize) < 65'536);

	markers.write_flag(normal_importance, L"reduce");

	try
	{
		// Reduce using parallel_for_each as long as element >= tile_size
		while (element_count >= TileSize && element_count % (TileSize * 2) == 0)
		{
			// halven extent, as the first reduction happens during saving to tile_static memory
			extent<1> halved_extent(element_count / 2);

			parallel_for_each(halved_extent.tile<TileSize>(),
				[=, &src, &dst](tiled_index<TileSize> tidx) restrict(amp)
			{
				// Use tile_static as a scratchpad memory.
				tile_static float tile_data[TileSize];

				unsigned local_idx = tidx.local[0];
				unsigned relative_idx = tidx.tile[0] * (TileSize * 2) + local_idx;
				// perform first reduction when populating tile_static memory
				tile_data[local_idx] = fast_math::fminf(src[relative_idx], src[relative_idx + TileSize]);
				tidx.barrier.wait();

				for (unsigned s = TileSize / 2; s > 0; s /= 2)
				{
					if (local_idx < s)
					{
						tile_data[local_idx] = fast_math::fminf(tile_data[local_idx], tile_data[local_idx + s]);
					}
					tidx.barrier.wait();
				}
				// Store the tile result in the global memory.
				if (local_idx == 0)
				{
					dst[tidx.tile[0]] = tile_data[0];
				}
			});
			// Update the sequence length, swap source with destination.
			element_count /= TileSize * 2;
			std::swap(src, dst);
		}
		// Perform any remaining reduction on the CPU.
		std::vector<float> result(element_count);

		// copy only part of array back to host, which contains the minimal elements (for performance reasons)
		copy(src.section(0, element_count), result.begin());

		// reduce all remaining tiles on the cpu and find the minimum element
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
the endvalue at rank 0.*/
template <const int TileSize>
void calculate_value_at_risk(std::vector<float>& host_end_values, const float initial_value,
	const float expected_return,
	const float volatility, const int trading_days, const int holding_period,
	const bool print_enabled = true)
{
	// time taken to initialize, no copying of data, entire implementation uses array instead of array_view to be able to measure copying times as well (cmp. p. 131 AMP book) 
	const auto start_initialize = the_amp_clock::now();
	array<float> gpu_end_values(host_end_values.size());
	gpu_end_values.accelerator_view.wait();
	const auto end_initialize = the_amp_clock::now();
	const auto elapsed_time_initialize = duration_cast<microseconds>(end_initialize - start_initialize).count();

	// first kernel: generate random paths
	generate_random_paths(initial_value, expected_return, volatility, trading_days, holding_period, gpu_end_values);

	gpu_end_values.accelerator_view.wait();
	auto const end_kernel_one = the_amp_clock::now();
	const auto elapsed_time_kernel_one = duration_cast<microseconds>(end_kernel_one - end_initialize).count();

	// write endvalues back to host for further investigation
	copy(gpu_end_values, host_end_values.begin());
	gpu_end_values.accelerator_view.wait();
	auto const end_copy = the_amp_clock::now();
	auto const elapsed_time_copying = duration_cast<microseconds>(end_copy - end_kernel_one).count();

	// second kernel: rearrange elements to obtain element at rank 0
	auto min_result = min_element<TileSize>(gpu_end_values);
	gpu_end_values.accelerator_view.wait();
	const auto end_kernel_two = the_amp_clock::now();
	const auto elapsed_time_kernel_two = duration_cast<microseconds>(end_kernel_two - end_copy).count();

	// total elapsed time. It can slightly differ from the individual times due to casting
	const auto elapsed_time_total = duration_cast<microseconds>(end_kernel_two - start_initialize).count();

	if (print_enabled)
	{
		/*
		// write measures to csv.
		file << elapsed_time_initialize << "," << elapsed_time_kernel_one << "," << elapsed_time_copying << "," <<
		elapsed_time_kernel_two << "," << elapsed_time_total << std::endl;
		*/

		std::cout << std::setfill('.');
		// stats 
		std::cout << "stats:" << std::endl;
		std::cout << std::setw(35) << std::left << "Initialize time (micro s):" << std::right << std::setw(8) <<
			elapsed_time_initialize << std::endl;
		std::cout << std::setw(35) << std::left << "Kernel one time (micro s):" << std::right << std::setw(8) <<
			elapsed_time_kernel_one << std::endl;
		std::cout << std::setw(35) << std::left << "copying time (micro s):" << std::right << std::setw(8) <<
			elapsed_time_copying << std::endl;
		std::cout << std::setw(35) << std::left << "Kernel two time (micro s):" << std::right << std::setw(8) <<
			elapsed_time_kernel_two << std::endl;
		std::cout << std::setw(35) << std::left << "Total time (micro s):" << std::right << std::setw(8) << elapsed_time_total
			<< std::endl << std::endl;


		// measures
		std::cout << "measures:" << std::endl;
		std::cout << std::setw(35) << std::left << "number of paths:" << std::right << std::setw(8) << host_end_values.size()
			<< std::endl;
		std::cout << std::setw(35) << std::left << "tile size:" << std::right << std::setw(8) << TileSize << std::endl;
		std::cout << std::setw(35) << std::left << "confidence level:" << std::right << std::setw(8) << "100 %" << std::endl;
		std::cout << std::setw(35) << std::left << "initial value:" << std::right << std::setw(8) << initial_value << std::
			endl;
		std::cout << std::setw(35) << std::left << "expected return:" << std::right << std::setw(8) << expected_return << std
			::endl;
		std::cout << std::setw(35) << std::left << "volatility:" << std::right << std::setw(8) << volatility << std::endl;
		std::cout << std::setw(35) << std::left << "trading days:" << std::right << std::setw(8) << trading_days << std::endl;
		std::cout << std::setw(35) << std::left << "holding period:" << std::right << std::setw(8) << holding_period << std::
			endl;
		const HANDLE h_console = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(h_console, 12);
		std::cout << std::setw(35) << std::left << "value at risk:" << std::right << std::setw(8) << min_result -
			initial_value << std::endl << std::endl;
		SetConsoleTextAttribute(h_console, 7);
	}
} // calculate_value_at_risk

/*
Helper function to avoid lazy initialization and just in time overhead (JIT) on first execution.
For details see: https://goo.gl/DPZuGU */
void warm_up()
{
	std::vector<float> paths(1024, 0);
	// run kernel with minimal dataset
	calculate_value_at_risk<32>(paths, 10.0f, 0.05f, 0.04f, 300, 10, false);
} // warm_up

/* This is wrapper function around calculate_value_at_risk. It  changing the tile_size dynamically
by using template parameters. The tilesize must be known at compile time. An approach similar
to this is suggested in the AMP book (see p. 96-97).
*/
void run(const unsigned& tile_size, std::vector<float>& paths, const float initial_value = 10.0f,
	const float expected_return = 0.05f, const float volatility = 0.04f, const int trading_days = 300,
	const int holding_period = 300)
{
	switch (tile_size)
	{
	case 2:
		calculate_value_at_risk<2>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 4:
		calculate_value_at_risk<4>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 8:
		calculate_value_at_risk<8>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 16:
		calculate_value_at_risk<16>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 32:
		calculate_value_at_risk<32>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 64:
		calculate_value_at_risk<64>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 128:
		calculate_value_at_risk<128>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 256:
		calculate_value_at_risk<256>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 512:
		calculate_value_at_risk<512>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	case 1024:
		calculate_value_at_risk<1024>(paths, initial_value, expected_return, volatility, trading_days, holding_period);
		break;
	default:
		assert(false);
	}
} // run

int main(int argc, char* argv[])
{
	try
	{
		TCLAP::CmdLine cmd("AMPMC", ' ', "1");

		// Define constraints
		PostiveValueConstraint positiveValueConstraint;
		PercentageConstraint percentageConstraint;
		DaysConstraint daysConstraint;
		PathConstraint pathConstraint;
		TileConstraint tileConstraint;

		// Define the arguments
		TCLAP::ValueArg<float> initial_value("i", "initial_value", "Initial value of the investment.", false,0.04f,&positiveValueConstraint);
		TCLAP::ValueArg<float> annual_return("r", "annual_return", "Annual return of the investment", false, 0.05f, &percentageConstraint);
		TCLAP::ValueArg<float> annual_volatility("v", "annual_volatility", "Annual volalitility of the investment",false,0.04f, &percentageConstraint);
		TCLAP::ValueArg<int> trading_days("t", "trading_days", "Annual trading days", false, 300, &daysConstraint);
		TCLAP::ValueArg<int> holding_period("d", "duration", "Duration of the investment", false, 300, &daysConstraint);
		TCLAP::ValueArg<int> paths("p", "paths", "Number of paths", false, 1'024, &pathConstraint);
		TCLAP::ValueArg<int> tile_size("x", "tile_size", "Size of tiles", false, 16, &tileConstraint);
		
		cmd.add(initial_value);
		cmd.add(annual_return);
		cmd.add(annual_volatility);
		cmd.add(trading_days);
		cmd.add(holding_period);
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
			annual_volatility.getValue(), trading_days.getValue(), holding_period.getValue());
		/*
		// validate output by printing it to csv
		file.open("validation.csv", std::ios::out);
		for (float f : path_vector)
			file << f << "," << std::endl;
		file.close();
		*/
	}
	catch (TCLAP::ArgException& e)
	{
		std::cout << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}


	/*
	query_amp_support();
	warm_up();

	// run simulation 100 times with all combinations

	file.open("measures.csv", std::ios::out);
	file << "ps,ts,i,Initialize time, kernel one time, copying time, kernel two time, total time" << std::endl;
	for (auto i(1); i <= 100; i++) {
		for (auto ps(1024); ps <= 524'288; ps *= 2) {
			for (auto ts(16); ts <= 1024; ts *= 2) {
					warm_up();
					file << ps << "," << ts << "," << i << ",";
					std::vector<float> paths(ps);
					run(ts, paths);
			}
		}
	}
	file.close();
	*/
	return 0;
} // main
