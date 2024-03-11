//
// Created by aske on 3/11/24.
//

#ifndef GEO_RT_INDEX_PTA_FACTORY_HPP
#define GEO_RT_INDEX_PTA_FACTORY_HPP

#include "factories/factory.hpp"
#include "types.hpp"
#include <vector>

class PointToAABBFactory : public Factory<OptixBuildInput>
{
private:
	std::unique_ptr<cuda_buffer> points_d;
	std::unique_ptr<cuda_buffer> aabb_d;
	size_t num_points;
//	const std::vector<Point>& points;
//	OptixAabb query;
public:
	explicit PointToAABBFactory(const std::vector<Point>& _points);
	std::unique_ptr<OptixBuildInput> Build() override;
	void SetQuery(OptixAabb _query);
	Point* GetPointsDevicePointer() const;
	size_t GetNumPoints() const;
};

#endif // GEO_RT_INDEX_PTA_FACTORY_HPP
