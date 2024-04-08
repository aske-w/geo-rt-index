//
// Created by aske on 4/5/24.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include "helpers/spatial_helpers.cuh"
#include "../../generators/aabb.cuh"
#include "../../generators/point.cuh"
#include <vector>

using namespace geo_rt_index::helpers;
using std::cout, std::to_string;

enum class Location : uint8_t
{
	Host = 1,
	Device = 2,
	Both = Location::Host | Location::Device
};

constexpr inline bool operator&(const Location rhs, const Location lhs)
{
	return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs);
}

template<Location location>
void Inside();

template<Location location>
void Boundary();

template<Location location>
void Outside();

__global__ static void ContainsWrapper(const Aabb& aabb, const Point& point, bool& result)
{
	result = SpatialHelpers::Contains(aabb, point);
}

TEST_CASE("Points on AABB edge are not contained in AABB CPU", "[CPU]")
{
	Boundary<Location::Host>();
}
TEST_CASE("Points on AABB edge are not contained in AABB GPU", "[GPU]")
{
	Boundary<Location::Device>();
}
TEST_CASE("Points on AABB edge are not contained in AABB Both", "")
{
	Boundary<Location::Both>();
}

template<Location location>
void Boundary()
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
	bool result = false;
	if (location & Location::Device)
	{
		bool temp_result = true;
		cuda_buffer bbox_d, point_d, result_d;
		bbox_d.alloc_and_upload<Aabb>({bbox});
		point_d.alloc_and_upload<Point>({point});
		result_d.alloc(sizeof(bool));
		ContainsWrapper<<<1,1>>>(*bbox_d.ptr<const Aabb>(), *point_d.ptr<const Point>(), *result_d.ptr<bool>());
		result_d.download(&temp_result, 1);
		CHECK_FALSE(temp_result);
		result |= temp_result;
	}
	if (location & Location::Host)
	{
		bool temp_result = true;
		temp_result = SpatialHelpers::Contains(bbox, point);
		CHECK_FALSE(temp_result);
		result |= temp_result;
	}

	REQUIRE_FALSE(result);
}

TEST_CASE("Points outside AABB are not contained in AABB CPU", "")
{
	Outside<Location::Host>();
}
TEST_CASE("Points outside AABB are not contained in AABB GPU", "")
{
	Outside<Location::Device>();
}
TEST_CASE("Points outside AABB are not contained in AABB", "")
{
	Outside<Location::Both>();
}

template<Location location>
void Outside()
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

	bool result = false;
	if (location & Location::Device)
	{
		bool temp_result = true;
		cuda_buffer bbox_d, point_d, result_d;
		bbox_d.alloc_and_upload<Aabb>({bbox});
		point_d.alloc_and_upload<Point>({point});
		result_d.alloc(sizeof(bool));
		ContainsWrapper<<<1,1>>>(*bbox_d.ptr<const Aabb>(), *point_d.ptr<const Point>(), *result_d.ptr<bool>());
		result_d.download(&temp_result, 1);
		CHECK_FALSE(temp_result);
		result |= temp_result;
	}
	if (location & Location::Host)
	{
		bool temp_result = true;
		temp_result = SpatialHelpers::Contains(bbox, point);
		CHECK_FALSE(temp_result);
		result |= temp_result;
	}

	REQUIRE_FALSE(result);
}


TEST_CASE("Points inside AABB are contained in AABB CPU", "[CPU]")
{
	Inside<Location::Host>();
}

TEST_CASE("Points inside AABB are contained in AABB GPU", "")
{
	Inside<Location::Device>();
}

TEST_CASE("Points inside AABB are contained in AABB Both", "")
{
	Inside<Location::Both>();
}

template<Location location>
void Inside()
{
	const float float_max = GENERATE(
	    1e10f,
	    1.f,
	    take(20, random(0.1f, 1.f)),
	    take(10, random(1.f, 10.f)),
	    take(5, random(10.f, 100.f)),
	    take(2, random(100.f, 1000.f)),
	    take(2, Catch::Generators::random(1000.f, 1e9f))
	);
	const float float_min = GENERATE_REF(std::nextafterf(0.f, 1.f), -float_max);
	Aabb bbox{float_min, float_min, float_max, float_max};

	const float x = GENERATE_REF(take(50, random(std::nextafterf(float_min, std::numeric_limits<float>::max()), std::nextafterf(float_max, std::numeric_limits<float>::lowest()))));
	const float y = GENERATE_REF(take(50, random(std::nextafterf(float_min, std::numeric_limits<float>::max()), std::nextafterf(float_max, std::numeric_limits<float>::lowest()))));
	Point point{x, y};

	bool result = true;
	if (location & Location::Device)
	{
		bool temp_result = false;
		cuda_buffer bbox_d, point_d, result_d;
		bbox_d.alloc_and_upload<Aabb>({bbox});
		point_d.alloc_and_upload<Point>({point});
		result_d.alloc(sizeof(bool));
		ContainsWrapper<<<1,1>>>(*bbox_d.ptr<const Aabb>(), *point_d.ptr<const Point>(), *result_d.ptr<bool>());
		result_d.download(&temp_result, 1);
		CHECK(temp_result);
		result &= temp_result;
	}
	if (location & Location::Host)
	{
		bool temp_result = false;
		temp_result = SpatialHelpers::Contains(bbox, point);
		CHECK(temp_result);
		result &= temp_result;
	}

	REQUIRE(result);
}