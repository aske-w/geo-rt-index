//
// Created by aske on 4/11/24.
//

#ifndef GEO_RT_INDEX_ARGPARSER_HPP
#define GEO_RT_INDEX_ARGPARSER_HPP

#include <memory>

namespace geo_rt_index
{
namespace helpers
{

enum class Distribution : uint8_t
{
	UNIFORM = 0,
	GAUSSIAN = 1
};

class ValueRange
{
public:
	const float low, high;
	explicit ValueRange(float, float);
};

class Args
{
public:
	const uint64_t num_points;
	const uint32_t num_queries;
	const uint8_t selectivity;
	const Distribution point_distribution;
	const ValueRange value_range;
};

class ArgParser
{
private:
	const int argc;
	const char** argv;
public:
	explicit ArgParser(const int, const char**);
	const Args Parse();
};

}
}

#endif // GEO_RT_INDEX_ARGPARSER_HPP
