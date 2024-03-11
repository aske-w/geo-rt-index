//
// Created by aske on 3/11/24.
//

#ifndef GEO_RT_INDEX_DEBUG_HELPERS_HPP
#define GEO_RT_INDEX_DEBUG_HELPERS_HPP

#include <cstdio>

namespace geo_rt_index
{
namespace helpers
{


#if DEBUG_PRINT
#define D_PRINT(...) printf(__VA_ARGS__)

#else
#define D_PRINT(...)

#endif

} // helpers
} // geo_rt_index
#endif // GEO_RT_INDEX_DEBUG_HELPERS_HPP
