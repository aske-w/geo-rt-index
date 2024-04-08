#ifndef LAUNCH_PARAMETERS_HPP
#define LAUNCH_PARAMETERS_HPP

#include "types.hpp"
#include <optix_types.h>

using geo_rt_index::types::Point;
using geo_rt_index::types::Aabb;

struct LaunchParameters
{
    OptixTraversableHandle traversable;
#if INDEX_TYPE == 1
	Point* points;
	const size_t num_points = 0;
#endif
	bool* result_d;
	uint32_t* hit_count;
	Aabb query_aabb;
};

#endif //LAUNCH_PARAMETERS_HPP
