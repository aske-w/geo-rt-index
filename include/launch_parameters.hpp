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
	bool* result_d;
//	uint32_t** hit_count;
	OptixAabb* queries;
};

#endif //LAUNCH_PARAMETERS_HPP
