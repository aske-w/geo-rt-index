//
// Created by aske on 3/11/24.
//

#ifndef GEO_RT_INDEX_DEBUG_HELPERS_HPP
#define GEO_RT_INDEX_DEBUG_HELPERS_HPP

#include <cstdio>

//template<typename... Args>
//static void d_print(const char * fmt, Args... args)
//{
//#if DEBUG_PRINT
//	printf(fmt, args...);
//#endif
//}

#if DEBUG_PRINT
#define D_PRINT(...) printf(__VA_ARGS__)

#else
#define D_PRINT(...)

#endif

#endif // GEO_RT_INDEX_DEBUG_HELPERS_HPP
