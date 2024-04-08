//
// Created by aske on 4/8/24.
//

#include "helpers/input_generator.hpp"
#include "types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <cuda_runtime.h>

using geo_rt_index::types::Aabb;

/**
 * Tests
 * Query & space
 * - query == space --> num total must be num aabb
 * - query > space
 * - query < space --> fail
 * num total > num aabb
 * num total == num aabb
 * num total < num aabb  --> fail
 * shuffle
 */

TEST_CASE("If query AABB == space AABB, num_total must be num_in_aabb","")
{
	auto minX = GENERATE(take(5, random(-1e8f, 1e8f)));
	auto minY = GENERATE(take(5, random(-1e8f, 1e8f)));
	auto maxX = GENERATE_COPY(take(5, filter([minX](float f) {return f > minX;},random(-1e8f, std::nextafterf(1e8f, std::numeric_limits<float>::max())))));
	auto maxY = GENERATE_COPY(take(5, filter([minY](float f) {return f > minY;},random(-1e8f, std::nextafterf(1e8f, std::numeric_limits<float>::max())))));
	Aabb query{minX, minY, maxX, maxY};
	Aabb space{minX, minY, maxX, maxY};
	auto total = GENERATE(take(5, random(1u, 1000u)));
	auto in_aabb = total;
	REQUIRE_NOTHROW(geo_rt_index::helpers::InputGenerator::Generate(query, space, total, in_aabb));
	in_aabb = GENERATE_COPY(take(5, filter([total](uint32_t i) { return i != total; }, random(1u, 1000u))));
	REQUIRE_THROWS(geo_rt_index::helpers::InputGenerator::Generate(query, space, total, in_aabb));
}