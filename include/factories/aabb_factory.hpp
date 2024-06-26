//
// Created by aske on 3/7/24.
//

#ifndef GEO_RT_INDEX_AABB_FACTORY_HPP
#define GEO_RT_INDEX_AABB_FACTORY_HPP

#include "factories/factory.hpp"
#include "helpers/cuda_buffer.hpp"

#include <bits/unique_ptr.h>
#include <optix_types.h>

namespace geo_rt_index
{
namespace factories
{

class AabbFactory : public Factory<OptixBuildInput> {
private:
	std::unique_ptr<helpers::cuda_buffer> aabbs_d;
public:
	AabbFactory();
	std::unique_ptr<OptixBuildInput> Build() override;
};

} // factories
} // geo_rt_index


#endif // GEO_RT_INDEX_AABB_FACTORY_HPP
