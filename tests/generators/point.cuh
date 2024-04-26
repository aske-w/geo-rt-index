#include "types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/catch_get_random_seed.hpp>

#include <random>

class RandomPointGenerator : public Catch::Generators::IGenerator<geo_rt_index::types::Point>
{
private:
	std::minstd_rand m_rand;
	std::uniform_real_distribution<float> x_dist;
	std::uniform_real_distribution<float> y_dist;
	geo_rt_index::types::Point current;
public:
	RandomPointGenerator(const float min_x, const float min_y, const float max_x, const float max_y)
	    : m_rand(Catch::getSeed()),
	      x_dist(min_x, max_x),
	      y_dist(min_y, max_y),
	      current(0, 0)
	{
		next();
	}

	bool next() override
	{
		current = geo_rt_index::types::Point{x_dist(m_rand), y_dist(m_rand)};
		return true;
	}

	const geo_rt_index::types::Point & get() const override
	{
		return current;
	}
};

Catch::Generators::GeneratorWrapper<geo_rt_index::types::Point> randomPoint(const float min_x, const float min_y, const float max_x, const float max_y)
{
	return Catch::Generators::GeneratorWrapper(new RandomPointGenerator{min_x, min_y, max_x, max_y});
}