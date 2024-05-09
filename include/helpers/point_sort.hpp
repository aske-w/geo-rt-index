//
// Created by aske on 5/9/24.
//

#ifndef GEO_RT_INDEX_POINT_SORT_HPP
#define GEO_RT_INDEX_POINT_SORT_HPP

#include "types/point_sorting.hpp"
#include "types/point.hpp"
#include <vector>

namespace geo_rt_index
{
namespace helpers
{

class PointSort
{
public:
	static void Sort(std::vector<geo_rt_index::types::Point>&, const geo_rt_index::types::PointSorting);
};

}
}

#endif // GEO_RT_INDEX_POINT_SORT_HPP
