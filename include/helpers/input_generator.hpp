//
// Created by aske on 3/22/24.
//

#ifndef GEO_RT_INDEX_INPUT_GENERATOR_HPP
#define GEO_RT_INDEX_INPUT_GENERATOR_HPP

#include <vector>
#include <memory>
#include "types.hpp"

namespace geo_rt_index
{
namespace helpers
{

class InputGenerator
{
public:
	static std::vector<geo_rt_index::types::Point> Generate(const geo_rt_index::types::Aabb& query_aabb,
	                                                        const geo_rt_index::types::Aabb& space_aabb,
	                                                        uint32_t num_total,
	                                                        uint32_t num_in_aabb,
	                                                        const bool shuffle = true);
};

} // helpers
} // geo_rt_index

#endif // GEO_RT_INDEX_INPUT_GENERATOR_HPP
