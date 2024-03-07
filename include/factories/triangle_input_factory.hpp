//
// Created by aske on 2/26/24.
//

#ifndef GEO_RT_INDEX_TRIANGLE_INPUT_FACTORY_HPP
#define GEO_RT_INDEX_TRIANGLE_INPUT_FACTORY_HPP

#include "cuda_buffer.hpp"
#include "factory.hpp"

#include <optix_types.h>

class TriangleFactory : public Factory<OptixBuildInput> {
private:
	std::unique_ptr<cuda_buffer> triangles_d;
public:
	TriangleFactory();
	std::unique_ptr<OptixBuildInput> Build() override;
};

#endif // GEO_RT_INDEX_TRIANGLE_INPUT_FACTORY_HPP
