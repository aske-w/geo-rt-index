//
// Created by aske on 2/26/24.
//

#ifndef GEO_RT_INDEX_CURVE_FACTORY_HPP
#define GEO_RT_INDEX_CURVE_FACTORY_HPP

#include "factory.hpp"
#include "helpers/cuda_buffer.hpp"

#include <optix_types.h>

namespace geo_rt_index
{
namespace factories
{

class CurveFactory : public Factory<OptixBuildInput> {
private:
	std::unique_ptr<helpers::cuda_buffer> curve_points_d;
	std::unique_ptr<helpers::cuda_buffer> curve_indices_d;
	std::unique_ptr<helpers::cuda_buffer> curve_widths_d;
public:
	CurveFactory();
	std::unique_ptr<OptixBuildInput> Build() override;
};

} // factories
} // geo_rt_index


#endif // GEO_RT_INDEX_CURVE_FACTORY_HPP
