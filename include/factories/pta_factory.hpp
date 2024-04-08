//
// Created by aske on 3/11/24.
//

#ifndef GEO_RT_INDEX_PTA_FACTORY_HPP
#define GEO_RT_INDEX_PTA_FACTORY_HPP

#include "factories/factory.hpp"
#include "types.hpp"
#include <vector>

namespace geo_rt_index
{
namespace factories
{

using geo_rt_index::types::Point;
using geo_rt_index::types::Aabb;

class PointToAABBFactory : public Factory<OptixBuildInput>
{
private:
	std::unique_ptr<helpers::cuda_buffer> points_d;
	size_t num_points;
//	const std::vector<Point>& points;
//	OptixAabb query;
public:
	std::unique_ptr<helpers::cuda_buffer> aabb_d;
	explicit PointToAABBFactory(const std::vector<Point>& _points);
	std::unique_ptr<OptixBuildInput> Build() override;
	void SetQuery(Aabb _query);
	Point* GetPointsDevicePointer() const;
	size_t GetNumPoints() const;
};

} // factories
} // geo_rt_index


#endif // GEO_RT_INDEX_PTA_FACTORY_HPP
