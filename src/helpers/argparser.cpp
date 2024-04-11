//
// Created by aske on 4/11/24.
//

#include "helpers/argparser.hpp"
#include <exception>

namespace geo_rt_index
{
namespace helpers
{

ValueRange::ValueRange(float _low, float _high) : low(_low), high(_high)
{
}

Args::Args(uint64_t _num_points, uint32_t _num_queries, uint8_t _selectivity, geo_rt_index::helpers::Distribution _dist,
           float _low, float _high)
	: num_points(_num_points), num_queries(_num_queries), selectivity(_selectivity), point_distribution(_dist),
      value_range(_low, _high)
{
}

ArgParser::ArgParser(const int _argc, const char** _argv) : argc(_argc), argv(_argv)
{
}

const Args ArgParser::Parse()
{
	throw std::runtime_error("Not implemnted");
}

}
}