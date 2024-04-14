#ifndef LAUNCH_PARAMETERS_HPP
#define LAUNCH_PARAMETERS_HPP

#include "types.hpp"
#include "types/aabb.hpp"

#include <optix_types.h>

struct LaunchParameters
{
    OptixTraversableHandle traversable;
#if INDEX_TYPE == 1
	geo_rt_index::Point* points;
	const size_t num_points = 0;
#endif
	bool* result_d;
	uint32_t* hit_count;
	geo_rt_index::types::Aabb query_aabb;
};

#endif //LAUNCH_PARAMETERS_HPP
