//
// Created by aske on 3/7/24.
//

#include "factories/aabb_factory.hpp"
#include <vector>

using std::make_unique;
using std::unique_ptr;
using std::vector;

using namespace geo_rt_index::factories;

AabbFactory::AabbFactory() : aabbs_d(std::move(make_unique<cuda_buffer>()))
{

}

class Number {
private:
	uint i = 0;
public:
	Number(int _i) : i(_i) { }
	constexpr float operator ++(int) {
		i += 1;
		return static_cast<float>(i);
	}
};

unique_ptr<OptixBuildInput> AabbFactory::Build()
{
	static const constexpr uint32_t flags[] = {
	    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
	};
	Number i = 1;
	vector<OptixAabb> aabbs {
//	    {i++,0,0,i++,1,1},
//	    {i++,0,0,i++,1,1},
	    {0,0,0,1,1,1},
	    {2,0,0,3,1,1},
	    {4,0,0,5,1,1},
//	    {4,0,0,5,1,1}
//	    {0, 1, 0, 1,2,1 },
//		{1,1,1, -1,-1,-1 }
	};
	aabbs_d->alloc_and_upload(aabbs);
	auto bi = make_unique<OptixBuildInput>();
	memset(&*bi, 0, sizeof(OptixBuildInput));
	bi->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

	auto& prim= bi->customPrimitiveArray;
	prim.aabbBuffers = (CUdeviceptr*) &(aabbs_d->raw_ptr);
	prim.numPrimitives = aabbs.size();
	prim.numSbtRecords = 1;
	prim.strideInBytes = sizeof(OptixAabb);
	prim.flags = flags;
	prim.sbtIndexOffsetBuffer = 0;
	return std::move(bi);
}