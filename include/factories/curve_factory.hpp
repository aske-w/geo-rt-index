//
// Created by aske on 2/26/24.
//

#ifndef GEO_RT_INDEX_CURVE_FACTORY_HPP
#define GEO_RT_INDEX_CURVE_FACTORY_HPP

#include "cuda_buffer.hpp"
#include "factory.hpp"

#include <optix_types.h>

class CurveFactory : public Factory<OptixBuildInput> {
private:
	std::unique_ptr<cuda_buffer> curve_points_d;
	std::unique_ptr<cuda_buffer> curve_indices_d;
	std::unique_ptr<cuda_buffer> curve_widths_d;
public:
	CurveFactory();
	std::unique_ptr<OptixBuildInput> Build() override;
};

#endif // GEO_RT_INDEX_CURVE_FACTORY_HPP
