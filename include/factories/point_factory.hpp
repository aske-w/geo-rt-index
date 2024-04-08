//
// Created by aske on 3/7/24.
//

#ifndef GEO_RT_INDEX_POINT_FACTORY_HPP
#define GEO_RT_INDEX_POINT_FACTORY_HPP

#include "factory.hpp"
#include "helpers/cuda_buffer.hpp"
#include "types.hpp"

namespace geo_rt_index
{
namespace factories
{

class PointFactory : public Factory<OptixBuildInput> {
private:
	std::unique_ptr<helpers::cuda_buffer> points_d;
	std::unique_ptr<std::vector<geo_rt_index::types::Point>> points;
public:
	PointFactory();
	std::unique_ptr<OptixBuildInput> Build() override;
};

} // factories
} // geo_rt_index


#endif // GEO_RT_INDEX_POINT_FACTORY_HPP
