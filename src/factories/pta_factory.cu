#include "factories/pta_factory.hpp"
#include "helpers/time.hpp"
#include <optix_types.h>

using std::make_unique;
using namespace geo_rt_index::factories;
using geo_rt_index::helpers::cuda_buffer;

using std::vector;

PointToAABBFactory::PointToAABBFactory(const vector<types::Point>& _points,
                                       const vector<OptixAabb>& _queries)
	: points_d(std::move(make_unique<cuda_buffer>())),
      queries_d(std::move(make_unique<cuda_buffer>())),
      num_points(_points.size()),
      num_queries(_queries.size())

{
	MEASURE_TIME("Uploading points to GPU",
		 points_d->alloc_and_upload(_points);
	);

	MEASURE_TIME("Uploading queries to GPU",
		 queries_d->alloc_and_upload(_queries);
	);
}

std::unique_ptr<OptixBuildInput> PointToAABBFactory::Build()
{
	static const constexpr uint32_t flags[] = {
	    OPTIX_GEOMETRY_FLAG_NONE
	};
	auto bi = make_unique<OptixBuildInput>();
	memset(&*bi, 0, sizeof(OptixBuildInput));
	bi->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

	auto& prim= bi->customPrimitiveArray;
	prim.aabbBuffers = (CUdeviceptr*) &(queries_d->raw_ptr);
	prim.numPrimitives = num_queries;
	prim.numSbtRecords = 1;
	prim.strideInBytes = sizeof(OptixAabb);
	prim.flags = flags;
	prim.sbtIndexOffsetBuffer = 0;
	return std::move(bi);
}

geo_rt_index::Point* PointToAABBFactory::GetPointsDevicePointer() const
{
	return reinterpret_cast<Point*>(this->points_d->raw_ptr);
}
OptixAabb* PointToAABBFactory::GetQueriesDevicePointer() const
{
	return reinterpret_cast<OptixAabb*>(this->queries_d->raw_ptr);
}
