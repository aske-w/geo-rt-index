//
// Created by aske on 3/28/24.
//

#ifndef GEO_RT_INDEX_TIME_HPP
#define GEO_RT_INDEX_TIME_HPP

#include <chrono>
#include "helpers/argparser.hpp"
#include <functional>
#include <vector>

#ifdef USE_MEASURE_TIME

static void MeasureTime(const char* msg, std::function<void(void)> subject)
{
	const auto begin = std::chrono::steady_clock::now();
	subject();
	const auto end = std::chrono::steady_clock::now();
	const std::chrono::duration<double> duration = end - begin;
	printf("%s: %.3fs.\n", msg, duration.count());
}

#define MEASURE_TIME(msg, ...) 					\
{                                               \
	MeasureTime(msg, [&]() { __VA_ARGS__; });	\
}

#else
#define MEASURE_TIME(msg, ...) __VA_ARGS__;

#endif



#endif // GEO_RT_INDEX_TIME_HPP
