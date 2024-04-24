#ifndef LAUNCH_PARAMETERS_HPP
#define LAUNCH_PARAMETERS_HPP

#include "types.hpp"
#include "types/aabb.hpp"

#include <optix_types.h>

struct LaunchParameters
{
    OptixTraversableHandle traversable;
	geo_rt_index::Point* points;
	const size_t num_points = 0;
	const size_t max_z = 0;
	bool* result_d;
	const uint32_t rays_per_thread = 1;
//	uint32_t** hit_count;
	OptixAabb* queries;
};

#endif //LAUNCH_PARAMETERS_HPP
