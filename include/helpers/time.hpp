//
// Created by aske on 3/28/24.
//

#ifndef GEO_RT_INDEX_TIME_HPP
#define GEO_RT_INDEX_TIME_HPP

#include "helpers/argparser.hpp"
#include "helpers/global_state.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <iostream>

static const std::string csv_header{"id,program,benchmark,distribution,lohi,warmup,modifier,num_repetitions,rays_per_thread,aabb_layer_type,num_queries,num_files,metric,duration"};

using duration = std::chrono::duration<double>;

namespace geo_rt_index
{
namespace helpers
{

inline static void PrintCSVHeader()
{
	std::cout << csv_header << '\n';
}

inline static void PrintCSV(const char* msg, const uint32_t value)
{
	const auto& arg_instance = geo_rt_index::helpers::Args::GetInstance();
	const auto& id = arg_instance.GetID();
	const std::string program{"geo-rt-index"}; // benchmark,distribution,lohi,warmup
	const auto& distribution = arg_instance.GetDistribution();
	const auto lo = arg_instance.GetLo();
	const auto hi = arg_instance.GetHi();
	const auto warmup = geo_rt_index::helpers::GlobalState::GetIsWarmup();
	const auto& benchmark = arg_instance.GetBenchmark();
	const auto modifier = arg_instance.GetModifier();
	const auto num_repetitions = arg_instance.GetRepetitions();
	const auto rays_per_thread =arg_instance.GetRaysPerThread();
	const auto layer_type = arg_instance.GetLayering();
	const auto num_queries = arg_instance.GetQueries().size();
	const auto num_files = arg_instance.GetFiles().size();
	const auto metric = msg;
	printf("\"%s\",\"%s\",\"%s\",\"%s\",%d%d,%hhu,%f,%hhu,%u,%hhu,%zu,%zu,\"%s\",%u\n",
	    id.c_str(),
	    program.c_str(),
	    benchmark.c_str(),
	    distribution.c_str(),
	    lo,
	    hi,
	    warmup,
	    modifier,
	    num_repetitions,
	    rays_per_thread,
	    static_cast<uint8_t>(layer_type),
	    num_queries,
	    num_files,
	    metric,
	    value
	);
}

inline static void PrintCSV(const char* msg, const duration& duration)
{
	const auto& arg_instance = geo_rt_index::helpers::Args::GetInstance();
	const auto& id = arg_instance.GetID();
	const std::string program{"geo-rt-index"}; // benchmark,distribution,lohi,warmup
	const auto& distribution = arg_instance.GetDistribution();
	const auto lo = arg_instance.GetLo();
	const auto hi = arg_instance.GetHi();
	const auto warmup = geo_rt_index::helpers::GlobalState::GetIsWarmup();
	const auto& benchmark = arg_instance.GetBenchmark();
	const auto modifier = arg_instance.GetModifier();
	const auto num_repetitions = arg_instance.GetRepetitions();
	const auto rays_per_thread =arg_instance.GetRaysPerThread();
	const auto layer_type = arg_instance.GetLayering();
	const auto num_queries = arg_instance.GetQueries().size();
	const auto num_files = arg_instance.GetFiles().size();
	const auto metric = msg;
	printf("\"%s\",\"%s\",\"%s\",\"%s\",%d%d,%hhu,%f,%hhu,%u,%hhu,%zu,%zu,\"%s\",%.4f\n",
	    id.c_str(),
	    program.c_str(),
	    benchmark.c_str(),
	    distribution.c_str(),
	    lo,
	    hi,
	    warmup,
	    modifier,
	    num_repetitions,
	    rays_per_thread,
	    static_cast<uint8_t>(layer_type),
	    num_queries,
	    num_files,
	    metric,
	    duration.count()
	);
}

} // helpers
} // geo_rt_index


#ifdef USE_MEASURE_TIME

static void MeasureTime(const char* msg, std::function<void(void)> subject)
{
	nvtxRangePushA(msg);
	const auto begin = std::chrono::steady_clock::now();
	subject();
	const auto end = std::chrono::steady_clock::now();
	const duration duration = end - begin;
//	printf("%s: %.3fs.\n", msg, duration.count());
	geo_rt_index::helpers::PrintCSV(msg, duration);
	nvtxRangePop();
}


#define MEASURE_TIME(msg, ...) 					\
{                                               \
	MeasureTime(msg, [&]() { __VA_ARGS__; });	\
}

#else
#define MEASURE_TIME(msg, ...) __VA_ARGS__;

#endif


#endif // GEO_RT_INDEX_TIME_HPP
