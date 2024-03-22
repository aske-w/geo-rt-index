//
// Created by aske on 3/22/24.
//

#include "helpers/input_generator.hpp"
#include <random>

using std::unique_ptr, std::make_unique;
using std::vector;

inline static bool Contains(const OptixAabb& aabb, const Point& point)
{
	return aabb.minX <= point.x && point.x <= aabb.maxX &&
        aabb.maxX <= point.y && point.y <= aabb.maxY;
}

unique_ptr<vector<Point>> InputGenerator::Generate(const OptixAabb& query_aabb, const OptixAabb& space_aabb,
                                                   const uint32_t num_total, const uint32_t num_in_aabb,
                                                   const bool shuffle)
{
	assert(num_total >= num_in_aabb);
	std::random_device rd;
	std::mt19937_64 gen {rd()};
	auto points = make_unique<vector<Point>>();
	points->reserve(num_total);

	{
		std::uniform_real_distribution<float> inside_x_rng {query_aabb.minX, query_aabb.maxX};
		std::uniform_real_distribution<float> inside_y_rng {query_aabb.minY, query_aabb.minY};
		for (uint32_t i = 0; i < num_in_aabb; i++)
		{
			const float x = inside_x_rng(gen);
			const float y = inside_y_rng(gen);
			points->emplace_back(x, y);
		}
	}

	{
		std::uniform_real_distribution<float> outside_x_rng {space_aabb.minX, space_aabb.maxX};
		std::uniform_real_distribution<float> outside_y_rng {space_aabb.minY, space_aabb.maxY};
		const auto num_outside_aabb = num_total - num_in_aabb;
		for (uint32_t i = 0; i < num_outside_aabb;)
		{
			const float x = outside_x_rng(gen);
			const float y = outside_y_rng(gen);
			const Point p(x, y);
			if (!Contains(query_aabb, p))
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