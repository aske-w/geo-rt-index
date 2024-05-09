//
// Created by aske on 5/9/24.
//

#ifndef GEO_RT_INDEX_POINT_SORTING_HPP
#define GEO_RT_INDEX_POINT_SORTING_HPP

#include <cstdint>

namespace geo_rt_index
{
namespace types
{

enum class PointSorting : uint8_t
{
	None = 0,
	X = 1,
	Y = 2,
	ZOrderCurve = 3,

	First = None,
	Last = ZOrderCurve
};

}
}

#endif // GEO_RT_INDEX_POINT_SORTING_HPP
