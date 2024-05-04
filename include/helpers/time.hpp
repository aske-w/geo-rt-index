//
// Created by aske on 3/28/24.
//

#ifndef GEO_RT_INDEX_TIME_HPP
#define GEO_RT_INDEX_TIME_HPP

#include "helpers/argparser.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <nvtx3/nvToolsExt.h>
#include <vector>

static const std::string csv_header{"id,modifier,num_repetitions,rays_per_thread,aabb_layer_type,num_queries,num_files,metric,duration"};

namespace geo_rt_index
{
namespace helpers
{

inline static void PrintCSVHeader()
{
	std::cout << csv_header << '\n';
}

} // helpers
} // geo_rt_index


#ifdef USE_MEASURE_TIME


using duration = std::chrono::duration<double>;

inline static void PrintCSV(const char* msg, const duration& duration)
{
	const auto& arg_instance = geo_rt_index::helpers::Args::GetInstance();
	const auto& id = arg_instance.GetID();
	const auto modifier = arg_instance.GetModifier();
	const auto num_repetitions = arg_instance.GetRepetitions();
	const auto rays_per_thread =arg_instance.GetRaysPerThread();
	const auto layer_type = arg_instance.GetLayering();
	const auto num_queries = arg_instance.GetQueries().size();
	const auto metric = msg;
	printf("\"%s\",%f,%u,%u,%hhu,%zu,\"%s\",%.3f\n",
	    id.c_str(),modifier,num_repetitions,rays_per_thread,static_cast<uint8_t>(layer_type),
	    num_queries,metric,duration.count());
}

static void MeasureTime(const char* msg, std::function<void(void)> subject)
{
	nvtxRangePushA(msg);
	const auto begin = std::chrono::steady_clock::now();
	subject();
	const auto end = std::chrono::steady_clock::now();
	const duration duration = end - begin;
//	printf("%s: %.3fs.\n", msg, duration.count());
	PrintCSV(msg, duration);
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
