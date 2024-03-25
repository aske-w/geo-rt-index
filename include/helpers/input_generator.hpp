//
// Created by aske on 3/22/24.
//

#ifndef GEO_RT_INDEX_INPUT_GENERATOR_HPP
#define GEO_RT_INDEX_INPUT_GENERATOR_HPP

#include <vector>
#include <memory>
#include "types.hpp"

class InputGenerator
{
public:
	static std::unique_ptr<std::vector<Point>> Generate(const Aabb& query_aabb, const Aabb& space_aabb, uint32_t num_total, uint32_t num_in_aabb, const bool shuffle = true);
};


#endif // GEO_RT_INDEX_INPUT_GENERATOR_HPP
