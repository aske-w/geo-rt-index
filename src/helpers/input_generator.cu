//
// Created by aske on 3/22/24.
//

#include "helpers/input_generator.hpp"
#include <random>
#include "helpers/spatial_helpers.cuh"

using namespace geo_rt_index;
using std::unique_ptr, std::make_unique;
using std::vector;
using std::uniform_real_distribution;
using helpers::SpatialHelpers;

unique_ptr<vector<Point>> InputGenerator::Generate(const Aabb& query_aabb, const Aabb& space_aabb,
                                                   const uint32_t num_total, const uint32_t num_in_aabb,
                                                   const bool shuffle)
{
	assert(num_total >= num_in_aabb);
	std::random_device rd;
//	std::mt19937_64 gen {rd()};
	std::mt19937_64 gen {1337};
	auto points = make_unique<vector<Point>>();
	points->reserve(num_total);
	uniform_real_distribution<float> rng{0, 1};

	{
		uniform_real_distribution<float> inside_x_rng {query_aabb.minX, query_aabb.maxX};
		uniform_real_distribution<float> inside_y_rng {query_aabb.minY, query_aabb.maxY};
		for (uint32_t i = 0; i < num_in_aabb; i++)
		{
			const float x = inside_x_rng(gen);
			const float y = inside_y_rng(gen);
			points->emplace_back(x, y);
		}
	}

//	std::cout << *points << '\n';

	{
		uniform_real_distribution<float> outside_x_rng {space_aabb.minX, space_aabb.maxX};
		uniform_real_distribution<float> outside_y_rng {space_aabb.minY, space_aabb.maxY};
		const auto num_outside_aabb = num_total - num_in_aabb;
		for (uint32_t i = 0; i < num_outside_aabb;)
		{
			const float x = outside_x_rng(gen);
			const float y = outside_y_rng(gen);
			const Point p(x, y);
			if (!SpatialHelpers::Contains(query_aabb, p))
			{
				points->push_back(std::move(p));
				i++;
			}
		}
	}

	if(shuffle)
	{
		std::shuffle(points->begin(), points->end(), gen);
	}

	return points;
}