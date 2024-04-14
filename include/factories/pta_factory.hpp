//
// Created by aske on 3/11/24.
//

#ifndef GEO_RT_INDEX_PTA_FACTORY_HPP
#define GEO_RT_INDEX_PTA_FACTORY_HPP

#include "factories/factory.hpp"
#include "types.hpp"
#include "types/aabb.hpp"

#include <vector>

namespace geo_rt_index
{
namespace factories
{

class PointToAABBFactory : public Factory<OptixBuildInput>
{
private:
	std::unique_ptr<helpers::cuda_buffer> points_d;
	std::unique_ptr<helpers::cuda_buffer> queries_d;
	const size_t num_points;
	const size_t num_queries;
public:
	explicit PointToAABBFactory(const std::vector<Point>&, const std::vector<OptixAabb>&);
	std::unique_ptr<OptixBuildInput> Build() override;
	Point* GetPointsDevicePointer() const;
	OptixAabb* GetQueriesDevicePointer() const;
};

} // factories
} // geo_rt_index


#endif // GEO_RT_INDEX_PTA_FACTORY_HPP
