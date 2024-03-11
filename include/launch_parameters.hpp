#ifndef LAUNCH_PARAMETERS_HPP
#define LAUNCH_PARAMETERS_HPP

#include "types.hpp"
#include <optix_types.h>

struct
    launch_parameters {
    OptixTraversableHandle traversable;
//	Triangle* curve_points_d;
	uint32_t* result_d;
};

#endif //LAUNCH_PARAMETERS_HPP
