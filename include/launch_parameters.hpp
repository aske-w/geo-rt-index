#ifndef LAUNCH_PARAMETERS_HPP
#define LAUNCH_PARAMETERS_HPP

#include "types.hpp"
#include "types/aabb.hpp"

#include <optix_types.h>

using geo_rt_index::types::Point;
using geo_rt_index::types::Aabb;

struct LaunchParameters
{
    OptixTraversableHandle traversable;
	geo_rt_index::types::Point* points;
	const size_t num_points = 0;
	const size_t max_z = 0;
	bool* result_d;
	const uint32_t rays_per_thread = 1;
	uint32_t* false_positive_count;
	OptixAabb* queries;
	const float ray_length;
	uint32_t* intersect_count;
};

#endif //LAUNCH_PARAMETERS_HPP
