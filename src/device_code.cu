#include <cuda_runtime.h>
#include "types.hpp"
#include "launch_parameters.hpp"
#include "optix_helpers.cuh"
#include <optix.h>

#include <limits>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" __constant__ launch_parameters params;


extern "C" __global__ void __closesthit__test() {
	printf("__closesthit__test\n");
    // do nothing
}


extern "C" __global__ void __miss__test() {
	printf("__miss__test\n");
    // do nothing
}


// this function is called for every potential ray-aabb intersection
extern "C" __global__ void __intersection__test() {
	const uint32_t primitive_id = optixGetPrimitiveIndex();
	printf("__intersection__test %u\n", primitive_id);
//	printf("Is frontface hit: %x ", optixIsFrontFaceHit());
//	printf("Is backface hit: %x ", optixIsBackFaceHit());
//	printf("result_count %u\n", params.result_count);
//	printf("result_d %llX\n", params.result_d);
//	printf("access %u\n", params.result_d[x]);
	auto x = optixGetPayload_0();
	params.result_d[x] = primitive_id;
//	printf("write");
	optixSetPayload_0(x + 1);
//	set_payload_32(primitive_id);
}


// this function is called for every reported (i.e. confirmed) ray-primitive intersection
extern "C" __global__ void __anyhit__test() {
	const uint32_t primitive_id = optixGetPrimitiveIndex();
	printf("__anyhit_test %d\n", primitive_id);
	set_payload_32(primitive_id);
	optixIgnoreIntersection();
}


// this is the entry point
extern "C" __global__ void __raygen__test() {
	printf("__raygen_test\n");
	constexpr const uint32_t ray_flags = 0;
	const constexpr float t_max= 100;
	const float3 origin {0.5,1.5,0.5};
	const float3 direction {t_max,1.5,0.5};
	uint32_t i0 = 0;
	optixTrace(params.traversable, origin, direction, 0, t_max, 0.0f, OptixVisibilityMask(255), ray_flags, 0, 0,
			   0, i0);
	printf("__raygen_test:%d\n",i0);
//	for(uint i = 0; i < 1000; i++)
//	{
//	}
//	for (float i = -1; i < 1.0f; i += 0.1f)
//		for (float j = -1; j < 1.0f; j += 0.1f)
//			for (float k = -1; k < 1.0f; k += 0.1f)
//			{
//				float3 direction {
//				    i,j,k
//				};
//				optixTrace(params.traversable, origin, direction, -10, 10, 100.0f, OptixVisibilityMask(255), ray_flags, 0, 0,
//				           0, i0);
////
//////				params.result_d[count++] = i0;
//			}
}
