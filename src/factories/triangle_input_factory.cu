//
// Created by aske on 2/26/24.
//

#include "factories/triangle_input_factory.hpp"

#include "cuda_buffer.hpp"
#include "types.hpp"
#include <optix_types.h>
#include <cstring>
#include <vector>

using std::unique_ptr;
using std::make_unique;
using std::vector;
using namespace geo_rt_index::factories;

TriangleFactory::TriangleFactory() : triangles_d(std::move(std::make_unique<cuda_buffer>())) { }

unique_ptr<OptixBuildInput> TriangleFactory::Build() {
	static const constexpr uint32_t flags[] = {
	    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
	};

	vector<Triangle> triangles {
	    {
	        {0, 0, 1},
	        {0, 1, 0},
	        {1, 0, 0},
	    }
	};
	triangles_d->alloc_and_upload(triangles);
	auto bi = make_unique<OptixBuildInput>();
	memset(&*bi, 0, sizeof(OptixBuildInput));
	bi->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	bi->triangleArray.numVertices         = Triangle::vertex_count();
	bi->triangleArray.vertexBuffers       = (CUdeviceptr*) &(triangles_d->raw_ptr);
	bi->triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
	bi->triangleArray.vertexStrideInBytes = sizeof(float3);
	bi->triangleArray.numIndexTriplets    = 0;
	bi->triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
	bi->triangleArray.preTransform        = 0;
	bi->triangleArray.flags               = flags;
	bi->triangleArray.numSbtRecords       = 1;
	return std::move(bi);
}