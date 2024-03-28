//
// Created by aske on 3/28/24.
//

#ifndef GEO_RT_INDEX_TIME_HPP
#define GEO_RT_INDEX_TIME_HPP

#include <chrono>

#ifdef USE_MEASURE_TIME

#define MEASURE_TIME(msg, ...) 																	\
{                                               \
	const auto begin = std::chrono::steady_clock::now(); 											\
	__VA_ARGS__;																	\
	const auto end = std::chrono::steady_clock::now();                  							\
	const auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); \
	D_PRINT("%s: %ld.%03ld s.\n", msg, total_time_ms / 1000, total_time_ms % 1000);              \
}

#else
#define MEASURE_TIME(msg, ...) __VA_ARGS__;

#endif



#endif // GEO_RT_INDEX_TIME_HPP
