//
// Created by aske on 3/22/24.
//

#include <future>
#include "helpers/input_generator.hpp"
#include <random>
#include "helpers/spatial_helpers.cuh"
#include "helpers/exception.hpp"
#include "helpers/general.hpp"

using namespace geo_rt_index;
using std::unique_ptr, std::make_unique;
using std::vector;
using std::uniform_real_distribution;
using helpers::SpatialHelpers;
using std::nextafterf, std::numeric_limits;
using geo_rt_index::helpers::ArgumentException;
using geo_rt_index::helpers::string_format;

static inline constexpr float NextAfter(const float f)
{
	return nextafterf(f, numeric_limits<float>::max());
}

static inline constexpr float PreviousBefore(const float f)
{
	return nextafterf(f, numeric_limits<float>::min());
}

static vector<Point> Worker(const Aabb& query_aabb, const Aabb& space_aabb, const uint32_t num, uint64_t seed)
{
	static thread_local std::mt19937_64 gen{seed}; // thread local because of https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
	vector<Point> points;
	points.reserve(num);
	{
		uniform_real_distribution<float> outside_x_rng {space_aabb.minX, space_aabb.maxX};
		uniform_real_distribution<float> outside_y_rng {space_aabb.minY, space_aabb.maxY};
		for (uint32_t i = 0; i < num;)
		{
			const float x = outside_x_rng(gen);
			const float y = outside_y_rng(gen);
			const Point p(x, y);
			if (!SpatialHelpers::Contains(query_aabb, p))
			{
				points.push_back(std::move(p));
				i++;
			}
		}
	}
	return points;
}

vector<Point> InputGenerator::Generate(const Aabb& query_aabb, const Aabb& space_aabb,
                                                   const uint32_t num_total, const uint32_t num_in_aabb,
                                                   const bool shuffle)
{
	if (num_total == 0)
	{
		throw ArgumentException(nameof(num_total), string_format("%s must be greater than zero", nameof(num_total)));
	}
	if (num_total < num_in_aabb)
	{
		throw ArgumentException{nameof(num_in_aabb), string_format("%s may not be greater than %s", nameof(num_in_aabb), nameof(num_total))};
	}
	std::random_device rd;
	const auto seed = rd();
	D_PRINT("InputGenerator seed: %d\n", seed);
	std::mt19937_64 gen{seed};
	auto points = vector<Point>();
	points.reserve(num_total);
	uniform_real_distribution<float> rng{0, 1};

	{
		uniform_real_distribution<float> inside_x_rng {NextAfter(query_aabb.minX), PreviousBefore(query_aabb.maxX)};
		uniform_real_distribution<float> inside_y_rng {NextAfter(query_aabb.minY), PreviousBefore(query_aabb.maxY)};
		for (uint32_t i = 0; i < num_in_aabb; i++)
		{
			const float x = inside_x_rng(gen);
			const float y = inside_y_rng(gen);
			points.emplace_back(x, y);
		}
	}

//	std::cout << *points << '\n';

	const auto num_outside_aabb = num_total - num_in_aabb;
	vector<std::future<vector<Point>>> futures;
	const uint32_t work_per_thread = 1 << 21; // 4.2 million
	uint32_t work_issued = 0;
	while(work_issued < num_outside_aabb)
	{
		auto handle = std::async(std::launch::async, Worker, query_aabb, space_aabb, std::min(work_per_thread, num_outside_aabb - work_issued), rd());
		futures.push_back(std::move(handle));
		work_issued += work_per_thread;
	}

	for(auto&& handle : futures)
	{
		auto v = handle.get();
		points.insert(points.end(), v.begin(), v.end());
	}

	if(shuffle)
	{
		std::shuffle(points.begin(), points.end(), gen);
	}

	return points;
}