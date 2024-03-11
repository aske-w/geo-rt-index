//
// Created by aske on 2/26/24.
//

#include "factories/curve_factory.hpp"
#include <optix_types.h>

using namespace geo_rt_index::factories;

CurveFactory::CurveFactory() : curve_points_d(std::move(std::make_unique<cuda_buffer>())),
      curve_indices_d(std::move(std::make_unique<cuda_buffer>())),
      curve_widths_d(std::move(std::make_unique<cuda_buffer>()))
{

}

std::unique_ptr<OptixBuildInput> CurveFactory::Build() {
//	static const constexpr uint32_t flags[] = {
//		OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
//	};
	std::vector<float3> curve_points {
	    { 1,1,1},
	    { 2,1,1},
	    { 1,2,1}
	};
	curve_points_d->alloc_and_upload(curve_points);
	std::vector<uint32_t> curve_index {
	    0,
	    1
	};
	curve_indices_d->alloc_and_upload(curve_index);
	std::vector<float> widths {
	    0,
	    0,
	    0
	};
	curve_widths_d->alloc_and_upload(widths);

	auto bi = std::make_unique<OptixBuildInput>();
	auto arr = &bi->curveArray;
	bi->type = OPTIX_BUILD_INPUT_TYPE_CURVES;
	arr->numPrimitives = 1;
	arr->flag = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
	arr->curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
	arr->numVertices = curve_points.size();
	arr->vertexBuffers = (CUdeviceptr*) &(curve_points_d->raw_ptr);
	arr->vertexStrideInBytes = sizeof(float3);
	arr->indexBuffer = curve_indices_d->cu_ptr();
	arr->indexStrideInBytes = sizeof(uint32_t);
	arr->widthBuffers = (CUdeviceptr*) &(curve_widths_d->raw_ptr);
	arr->widthStrideInBytes = sizeof(float);
	return std::move(bi);
}