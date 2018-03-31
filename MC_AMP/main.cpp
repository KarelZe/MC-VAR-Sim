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
void calculate_value_at_risk() restrict(amp) {
	// todo: implementation using radix select.
} // calculate_value_at_risk

/* This is the cpu implementation of calculate_value_at_risk */
void calculate_value_at_risk_cpu(std::vector<float>& pathVector, const float confidenceLevel) {
	const unsigned int RANK = static_cast<unsigned int>(NUMBER_OF_PATHS * (1 - confidenceLevel));
	// uses median of medians algorithm with complexity O(n) to rearrange results
	std::nth_element(pathVector.begin(), pathVector.begin() + RANK, pathVector.end());
	// print value at risk
	std::cout << "Value at risk at " << HOLDING_PERIOD << " days with " << confidenceLevel * 100 <<
		" % confidence: " << (pathVector.at(RANK) - INITIAL_VALUE) << " GPB (with - being risk and + being chance)" << std::endl;
}

/* This function calculates random paths using geometric brownian motion (GBM) for a given holding period. For
details on geometric brownian motion see: https://goo.gl/lrCeLJ.
*/
void generate_random_paths(const unsigned int seed, const int size, const float initialValue, const float expectedReturn, const float volatility, const float tradingDays, std::vector<float>& endValues) {
	const unsigned int RANK = 1;

	// validate that given input is optimal
	static_assert((HOLDING_PERIOD % 2 == 0), "The holding period must be a multiple of two.");

	/* todo: find out what extent is best for tinymyt_collection, large numbers lead to crash of program probably due
	/ to memory limitations. If solved change auto t to use global idx. Small tiny collection delivers falsy results
	meaning that a higher number of paths won't deliver a higher accurancy.*/
	const extent<RANK> tinyE(65'536);
	const tinymt_collection<RANK> randCollection(tinyE, seed);

	extent<RANK> e(size);
	array_view<float> endvaluesAv(e, endValues);

	// do not copy data to accelerator
	endvaluesAv.discard_data();

	// start clock for GPU version after array allocation
	the_amp_clock::time_point start = the_amp_clock::now();

	// wrap parallel_for_each in try catch to provide feedback on runtime exceptions
	try {
		parallel_for_each(endvaluesAv.extent, [=](index<1>idx) restrict(amp) {

			auto t = randCollection[idx % 65'536];

			float s(0.0f);
			float prevS(initialValue);

			// see https://goo.gl/Rb394n for rationelle behind modifying drift and volatility.
			// scale drift to timestep
			const float dailyDrift = expectedReturn / tradingDays;
			// scale volatility to timestep. Volatility scales with square root of time.
			// Use rsqrt for performance reasons (See Chapter 7 AMP-Book)
			const float dailyVolatility = volatility * fast_math::rsqrtf(tradingDays);
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
	const unsigned int seed(7859);
	auto i(1024);
	for (i; i <= 1'048'576; i *= 2) {
		// create vector holding all paths
		std::vector<float> pathVector(i, 0);

		std::cout << i <<", ";

		generate_random_paths(seed, i, INITIAL_VALUE, EXPECTED_RETURN, VOLATILITY, TRADING_DAYS, pathVector);
		/*
		// print the first view results for validation
		for (auto i(0); i < 10; i++) {
			std::cout << std::fixed << std::setprecision(5) << pathVector[i] << std::endl;
		}
		calculate_value_at_risk_cpu(pathVector, CONFIDENCE_LEVEL);
		*/
	}

	return 0;
} // main

// radix select implemntation starts here...
 // adapted from https://archive.codeplex.com/?p=ampalgorithms originial implementation by Ade Miller

template<typename T, int key_bit_width>
inline int radix_key_value(const float value, const unsigned key_idx) restrict(amp, cpu)
{
	const T mask = (1 << key_bit_width) - 1;
	return (value >> (key_idx * key_bit_width)) & mask;
}

// functions to convert float to uint back and forwards. Radix sort requires whole numbers for sorting
inline unsigned int convert_to_uint(const float& value) restrict(amp, cpu)
{
	const unsigned mask = -static_cast<int>(reinterpret_cast<const unsigned&>(value) >> 31) | 0x80000000;
	return reinterpret_cast<const unsigned&>(value) ^ mask;
}
// function to convert uint back to float
inline float convert_from_uint(const unsigned& value) restrict(amp, cpu)
{
	unsigned v = value ^ (((value >> 31) - 1) | 0x80000000);
	return reinterpret_cast<float&>(v);
}

// function to initialize bins
inline void initialize_bins(float* const bin_data, const int bin_count) restrict(amp)
{
	for (int b = 0; b < bin_count; ++b)
	{
		bin_data[b] = float(0);
	}
}

template <typename T, int tile_size, int tile_key_bit_width>
void radix_sort_tile_by_key(T* const tile_data, const int data_size, concurrency::tiled_index<tile_size> tidx, const int key_idx) restrict(amp)
{
	const unsigned bin_count = 1 << tile_key_bit_width;
	const int gidx = tidx.global[0];
	const int tlx = tidx.tile[0];
	const int idx = tidx.local[0];

	// Increment histogram bins for each element.

	tile_static unsigned long tile_radix_values[tile_size];
	tile_radix_values[idx] = pack_byte(1, _details::radix_key_value<T, tile_key_bit_width>(tile_data[idx], key_idx));
	tidx.barrier.wait_with_tile_static_memory_fence();

	tile_static unsigned long histogram_bins_scan[bin_count];
	if (idx == 0)
	{
		// Calculate histogram of radix values. Don't add values that are off the end of the data.
		unsigned long global_histogram = 0;
		const int tile_data_size = amp_algorithms::min<int>()(tile_size, (data_size - (tlx * tile_size)));
		for (int i = 0; i < tile_data_size; ++i)
		{
			global_histogram += tile_radix_values[i];
		}

		// Scan to get offsets for each histogram bin.

		histogram_bins_scan[0] = 0;
		for (int i = 1; i < bin_count; ++i)
		{
			histogram_bins_scan[i] = unpack_byte(global_histogram, i - 1) + histogram_bins_scan[i - 1];
		}
	}
	tidx.barrier.wait_with_tile_static_memory_fence();

	_details::scan_tile_exclusive<tile_size>(tile_radix_values, tidx, amp_algorithms::plus<unsigned long>());

	// Shuffle data into sorted order.

	T tmp = tile_data[idx];
	tidx.barrier.wait_with_tile_static_memory_fence();
	if (gidx < data_size)
	{
		const int rdx = _details::radix_key_value<T, tile_key_bit_width>(tmp, key_idx);
		unsigned long dest_idx = histogram_bins_scan[rdx] + unpack_byte(tile_radix_values[idx], rdx);
		tile_data[dest_idx] = tmp;
	}
}

template <typename T, int tile_size, int key_bit_width, int tile_key_bit_width = 2>
void radix_sort_by_key(const concurrency::accelerator_view& accl_view, const concurrency::array_view<T>& input_view, concurrency::array_view<T>& output_view, const int key_idx)
{
	static const unsigned type_width = sizeof(T) * CHAR_BIT;
	static const int bin_count = 1 << key_bit_width;

	static_assert((tile_size <= 256), "The tile size must be less than or equal to 256.");
	static_assert((key_bit_width >= 1), "The radix bit width must be greater than or equal to one.");
	static_assert((tile_size >= bin_count), "The tile size must be greater than or equal to the radix key bin count.");
	static_assert((type_width % key_bit_width == 0), "The sort key width must be divisible by the type width.");
	static_assert((key_bit_width % tile_key_bit_width == 0), "The key bit width must be divisible by the tile key bit width.");
	static_assert(tile_key_bit_width <= 2, "Only tile key bin widths of two or less are supported.");

	const concurrency::tiled_extent<tile_size> compute_domain = output_view.get_extent().tile<tile_size>().pad();
	const int tile_count = std::max(1u, compute_domain.size() / tile_size);

	concurrency::array<int, 2> per_tile_rdx_offsets(concurrency::extent<2>(tile_count, bin_count), accl_view);
	concurrency::array<int> global_rdx_offsets(bin_count, accl_view);
	concurrency::array<int, 1> tile_histograms(concurrency::extent<1>(bin_count * tile_count), accl_view);

	amp_algorithms::fill(accl_view, global_rdx_offsets.section(0, bin_count), 0);

	concurrency::parallel_for_each(accl_view, compute_domain, [=, &per_tile_rdx_offsets, &global_rdx_offsets, &tile_histograms](concurrency::tiled_index<tile_size> tidx) restrict(amp)
	{
		const int gidx = tidx.global[0];
		const int tlx = tidx.tile[0];
		const int idx = tidx.local[0];
		tile_static unsigned tile_data[tile_size];
		tile_static int per_thread_rdx_histograms[tile_size][bin_count];

		// Initialize histogram bins and copy data into tiles.
		initialize_bins(per_thread_rdx_histograms[idx], bin_count);
		tile_data[idx] = convert_to_uint<T>(padded_read(input_view, gidx));

		// Increment radix bins for each element on each tile.
		if (gidx < input_view.extent[0])
		{
			per_thread_rdx_histograms[idx][_details::radix_key_value<unsigned, key_bit_width>(tile_data[idx], key_idx)]++;
		}
		tidx.barrier.wait_with_tile_static_memory_fence();

		// First bin_count threads per tile collapse thread values to create the tile histogram.
		if (idx < bin_count)
		{
			for (int i = 1; i < tile_size; ++i)
			{
				per_thread_rdx_histograms[0][idx] += per_thread_rdx_histograms[i][idx];
			}
		}
		tidx.barrier.wait_with_tile_static_memory_fence();

		// First bin_count threads per tile increment counts for global histogram and copies tile histograms to global memory.
		if (idx < bin_count)
		{
			concurrency::atomic_fetch_add(&global_rdx_offsets[idx], per_thread_rdx_histograms[0][idx]);
		}



		// Exclusive scan the tile histogram to calculate the per-tile offsets.
		if (idx < bin_count)
		{
			tile_histograms[(idx * tile_count) + tlx] = per_thread_rdx_histograms[0][idx];
		}
		tidx.barrier.wait_with_tile_static_memory_fence();
		_details::scan_tile_exclusive<tile_size>(per_thread_rdx_histograms[0], tidx, amp_algorithms::plus<unsigned>());

		if (idx < bin_count)
		{
			per_tile_rdx_offsets[tlx][idx] = per_thread_rdx_histograms[0][idx];
		}
	});

	concurrency::parallel_for_each(accl_view, compute_domain, [=, &global_rdx_offsets, &tile_histograms](concurrency::tiled_index<tile_size> tidx) restrict(amp)
	{
		const int gidx = tidx.global[0];
		const int idx = tidx.local[0];

		// Calculate global radix offsets from the global radix histogram. All tiles do this but only the first one records the result.
		tile_static int scan_data[tile_size];
		scan_data[idx] = (idx < bin_count) ? global_rdx_offsets[idx] : 0;
		tidx.barrier.wait_with_tile_static_memory_fence();

		_details::scan_tile_exclusive<tile_size>(scan_data, tidx, amp_algorithms::plus<unsigned>());

		if (gidx < bin_count)
		{
			global_rdx_offsets[gidx] = scan_data[gidx];
		}
	});

	concurrency::array_view<int, 1> tile_histograms_vw(tile_histograms);
	scan_exclusive(tile_histograms_vw, tile_histograms_vw);

	concurrency::parallel_for_each(accl_view, compute_domain, [=, &per_tile_rdx_offsets, &tile_histograms, &global_rdx_offsets](concurrency::tiled_index<tile_size> tidx) restrict(amp)
	{
		const int gidx = tidx.global[0];
		const int tlx = tidx.tile[0];
		const int idx = tidx.local[0];


		// Sort elements within each tile.
		tile_static unsigned tile_data[tile_size];
		tile_data[idx] = convert_to_uint<T>(padded_read(input_view, gidx));

		tidx.barrier.wait_with_tile_static_memory_fence();

		const int keys_per_tile = (key_bit_width / tile_key_bit_width);
		for (int k = (keys_per_tile * key_idx); k < (keys_per_tile * (key_idx + 1)); ++k)
		{
			_details::radix_sort_tile_by_key<unsigned, tile_size, tile_key_bit_width>(tile_data, input_view.extent[0], tidx, k);
		}
		tidx.barrier.wait_with_tile_static_memory_fence();

		// Dump sorted per-tile data, sorted_per_tile

		// Move tile sorted elements to global destination.

		const int rdx = _details::radix_key_value<unsigned, key_bit_width>(tile_data[idx], key_idx);
		const int dest_gidx =
			idx -
			per_tile_rdx_offsets[tlx][rdx] +
			segment_exclusive_scan(tile_histograms_vw, tile_count, (rdx * tile_count) + tlx) +
			global_rdx_offsets[rdx];

		//output_view[gidx] = dest_gidx;                                                                                            // Dump destination indices, dest_gidx

		if (gidx < input_view.extent[0])
		{
			output_view[dest_gidx] = convert_from_uint<T>(tile_data[idx]);
		}
	});
}

template <typename T, int tile_size, int key_bit_width, int tile_key_bit_width = 2>
void radix_sort(const concurrency::accelerator_view& accl_view, concurrency::array_view<T>& input_view, concurrency::array_view<T>& output_view)
{
	static const int key_count = bit_count<T>() / key_bit_width;

	for (int key_idx = 0; key_idx < key_count; ++key_idx)
	{
		_details::radix_sort_by_key<T, tile_size, key_bit_width, tile_key_bit_width>(accl_view, input_view, output_view, key_idx);
		std::swap(output_view, input_view);
	}
	std::swap(input_view, output_view);
}