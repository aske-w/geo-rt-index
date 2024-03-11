//
// Created by aske on 3/7/24.
//

#ifndef GEO_RT_INDEX_POINT_FACTORY_HPP
#define GEO_RT_INDEX_POINT_FACTORY_HPP

#include "types.hpp"
#include "cuda_buffer.hpp"
#include "factory.hpp"
#include "types.hpp"

class PointFactory : public Factory<OptixBuildInput> {
private:
	std::unique_ptr<cuda_buffer> points_d;
	std::unique_ptr<std::vector<Point>> points;
public:
	PointFactory();
	std::unique_ptr<OptixBuildInput> Build() override;
};

#endif // GEO_RT_INDEX_POINT_FACTORY_HPP
