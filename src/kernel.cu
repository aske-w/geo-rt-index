#include "helpers/spatial_helpers.cuh"
#include "kernel.cuh"
#include "launch_parameters.hpp"

#include <helpers/debug_helpers.hpp>

__global__ void KernelFilter(const LaunchParameters* const params)
{
	const auto block_id = blockIdx.y * gridDim.x + blockIdx.x;
	const auto thread_id = block_id * blockDim.x + threadIdx.x;
	const auto point_id = thread_id;
	const auto primitive_id = blockIdx.z;
	const auto contained = geo_rt_index::helpers::SpatialHelpers::Contains(params->queries[primitive_id], params->points[point_id]);
	params->result_d[(primitive_id * params->num_points) + point_id] = contained;
}