//
// Created by aske on 4/5/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include "helpers/spatial_helpers.cuh"
#include "../../generators/aabb.cuh"
#include "../../generators/point.cuh"

using namespace geo_rt_index::helpers;
using std::cout, std::to_string;

static inline auto FloatMinMaxRand()
{
	return Catch::Generators::random(std::numeric_limits<float>::lowest(),std::numeric_limits<float>::max());
}

TEST_CASE("Points on AABB edge are not contained in AABB", "")
{
	const float float_max = GENERATE(
	    1e10f,
	    1.f,
	    take(1000, random(0.1f, 1.f)),
	    take(1000, random(1.f, 10.f)),
	    take(1000, random(10.f, 100.f)),
	    take(1000, random(100.f, 1000.f)),
	    take(1000, Catch::Generators::random(1000.f, 1e9f))
	);
	const float float_min = GENERATE_REF(0.f, -float_max);
	Aabb bbox{float_min, float_min, float_max, float_max};
	auto point = GENERATE_REF(
	    Point{float_min, float_min},
	    Point{float_min, float_max},
	    Point{float_max, float_min},
	    Point{float_max, float_max}
 	);
	REQUIRE_FALSE(SpatialHelpers::Contains(bbox, point));
}

TEST_CASE("Points outside AABB are not contained in AABB", "")
{
	const float float_max = GENERATE(
	    1e10f,
	    1.f,
	    take(200, random(0.1f, 1.f)),
	    take(100, random(1.f, 10.f)),
	    take(20, random(10.f, 100.f)),
	    take(10, random(100.f, 1000.f)),
	    take(10, Catch::Generators::random(1000.f, 1e9f))
	);
	const float float_min = GENERATE_REF(0.f, -float_max);
	Aabb bbox{float_min, float_min, float_max, float_max};
	auto x = GENERATE(take(100, random(0.1f, 1.f)), take(20, random(1.f, 1e4f)));
	auto point = GENERATE_REF(
	    Point{float_min - x, float_min},
	    Point{float_min, float_min - x},
	    Point{float_min - x, float_min - x},

	    Point{float_min - x, float_max},
	    Point{float_min, float_max + x},
	    Point{float_min - x, float_max + x},

	    Point{float_max + x, float_min},
	    Point{float_max, float_min - x},
	    Point{float_max + x, float_min - x},

	    Point{float_max + x, float_max},
	    Point{float_max, float_max + x},
	    Point{float_max + x, float_max + x}
	);
	REQUIRE_FALSE(SpatialHelpers::Contains(bbox, point));
}


TEST_CASE("Inside test", "")
{
	const constexpr float float_max = 1e8;
	const constexpr float float_min = -1 * float_max;
	const float min_next = std::nextafterf(float_min, 1.f);
	const float max_before = std::nextafterf(float_max, -1.f);
	Aabb bbox{float_min, float_min, float_max, float_max};
	auto point = GENERATE_REF(take(100, randomPoint(min_next, min_next, max_before, max_before)));
	REQUIRE(SpatialHelpers::Contains(bbox, point));
}