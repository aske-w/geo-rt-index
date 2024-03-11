//
// Created by aske on 3/7/24.
//

#include "factories/point_factory.hpp"
#include <vector>

using std::unique_ptr;
using std::make_unique;
using std::move;
using std::vector;

using namespace geo_rt_index::factories;

PointFactory::PointFactory() : points_d(move(make_unique<cuda_buffer>()))
{
	this->points = make_unique<vector<Point>>(vector<Point>{
//	    {0,0},
	    {1, 1},
	    {2, 1},
//	    {1, 2},
	    {3, 3},
	    {4, 3},
	    {5, 3},
	    {6, 3},
	});
}

std::unique_ptr<OptixBuildInput> PointFactory::Build()
{
	static const constexpr uint32_t flags[] = {
//	    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
	    OPTIX_GEOMETRY_FLAG_NONE
	};
	auto bi = make_unique<OptixBuildInput>();
	memset(&*bi, 0, sizeof(OptixBuildInput));
	vector<Triangle> transformed;
	for (auto&& p : *points)
	{
		transformed.push_back(p.ToTriangle());
	}

	this->points_d->alloc_and_upload(transformed);
	bi->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	bi->triangleArray.numVertices         = 3 * transformed.size();
	bi->triangleArray.vertexBuffers       = (CUdeviceptr*) &(points_d->raw_ptr);
	bi->triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
	bi->triangleArray.vertexStrideInBytes = sizeof(float3);
	bi->triangleArray.numIndexTriplets    = 0;
	bi->triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
	bi->triangleArray.preTransform        = 0;
	bi->triangleArray.flags               = flags;
	bi->triangleArray.numSbtRecords       = 1;
	return std::move(bi);
}