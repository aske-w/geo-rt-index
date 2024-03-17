#include "factories/pta_factory.hpp"

using std::make_unique;
using namespace geo_rt_index::factories;
using geo_rt_index::helpers::cuda_buffer;

PointToAABBFactory::PointToAABBFactory(const std::vector<Point>& _points)
	: points_d(std::move(make_unique<cuda_buffer>())),
      aabb_d(std::move(make_unique<cuda_buffer>()))

{
	points_d->alloc_and_upload(_points);
	num_points = _points.size();
}

void PointToAABBFactory::SetQuery(OptixAabb query)
{
	if (aabb_d->raw_ptr != nullptr)
		aabb_d->free();

	aabb_d->alloc_and_upload<OptixAabb>({query});
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
	prim.aabbBuffers = (CUdeviceptr*) &(aabb_d->raw_ptr);
	prim.numPrimitives = 1;
	prim.numSbtRecords = 1;
	prim.strideInBytes = sizeof(OptixAabb);
	prim.flags = flags;
	prim.sbtIndexOffsetBuffer = 0;
	return std::move(bi);
}

Point* PointToAABBFactory::GetPointsDevicePointer() const
{
	return reinterpret_cast<Point*>(this->points_d->raw_ptr);
}

size_t PointToAABBFactory::GetNumPoints() const
{
	return this->num_points;
}
