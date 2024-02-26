#include <cuda_runtime.h>
#include "types.hpp"
#include "launch_parameters.hpp"
#include "optix_helpers.cuh"
#include <optix.h>

#include <cstdint>
#include <cuda_runtime.h>

extern "C" __constant__ launch_parameters params;


extern "C" __global__ void __closesthit__test() {
    // do nothing
}


extern "C" __global__ void __miss__test() {
    // do nothing
}


// this function is called for every potential ray-aabb intersection
extern "C" __global__ void __intersection__test() {
	const uint32_t primitive_id = optixGetPrimitiveIndex();
	set_payload_32(primitive_id);
}


// this function is called for every reported (i.e. confirmed) ray-primitive intersection
extern "C" __global__ void __anyhit__test() {

	const uint32_t primitive_id = optixGetPrimitiveIndex();
	set_payload_32(primitive_id);
}


// this is the entry point
extern "C" __global__ void __raygen__test() {
	constexpr const uint32_t ray_flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
	float3 origin {0, 0, 0};
	float3 direction {1.2, 1.2, 1};
	uint32_t i0 = -1;
	optixTrace(params.traversable,
		origin,
		direction,
	           0,
	           5,
	           0.0f,
	           OptixVisibilityMask(255),
	           ray_flags,
	           0,
	           0,
	           0,
	           i0
   	);

	params.result_d[0] = i0;
}
