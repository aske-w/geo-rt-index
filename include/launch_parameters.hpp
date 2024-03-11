#ifndef LAUNCH_PARAMETERS_HPP
#define LAUNCH_PARAMETERS_HPP

#include "types.hpp"
#include <optix_types.h>

struct LaunchParameters
{
    OptixTraversableHandle traversable;
#if INDEX_TYPE == 1
	Point* points;
	const size_t num_points = 0;
#endif
	uint32_t* result_d;
};

#endif //LAUNCH_PARAMETERS_HPP
