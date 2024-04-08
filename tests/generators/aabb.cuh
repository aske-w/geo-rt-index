//
// Created by aske on 4/5/24.
//

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/catch_get_random_seed.hpp>

#include <random>

class RandomAabbGenerator : public Catch::Generators::IGenerator<geo_rt_index::types::Aabb>
{
private:
	std::minstd_rand m_rand;
	std::uniform_real_distribution<float> x_dist;
	std::uniform_real_distribution<float> y_dist;
	geo_rt_index::types::Aabb current;
public:
	RandomAabbGenerator(float min_x, float min_y, float max_x, float max_y)
	    : m_rand(Catch::getSeed()),
	      x_dist(min_x, max_x),
	      y_dist(min_y, max_y),
	      current(0,0,0,0)
	{
		next();
	}

	bool next() override
	{
		auto x_1 = x_dist(m_rand);
		auto x_2 = x_dist(m_rand);
		auto y_1 = y_dist(m_rand);
		auto y_2 = y_dist(m_rand);

		auto x_min = std::min(x_1, x_2);
		auto x_max = std::max(x_1, x_2);
		auto y_min = std::min(y_1, y_2);
		auto y_max = std::max(y_1, y_2);

		current = geo_rt_index::types::Aabb{x_min, y_min, x_max, y_max};
		return true;
	}
	const geo_rt_index::types::Aabb & get() const override
	{
		return current;
	}
};

Catch::Generators::GeneratorWrapper<geo_rt_index::types::Aabb> randomAabb(float min_x, float min_y, float max_x, float max_y)
{
	return Catch::Generators::GeneratorWrapper(new RandomAabbGenerator{min_x, min_y, max_x, max_y});
}