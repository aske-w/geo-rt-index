//
// Created by aske on 4/11/24.
//

#include "helpers/argparser.hpp"
#include <exception>
#include <string>

namespace geo_rt_index
{
namespace helpers
{

using std::stoi, std::stoll, std::stof;

ValueRange::ValueRange(float _low, float _high) : low(_low), high(_high)
{
}
//Args::Args() : value_range(0,0)
//{
//}
//Args::Args(uint64_t _num_points, uint32_t _num_queries, uint8_t _selectivity, geo_rt_index::helpers::Distribution _dist,
//           float _low, float _high)
//	: num_points(_num_points), num_queries(_num_queries), selectivity(_selectivity), point_distribution(_dist),
//      value_range(_low, _high)
//{
//}

ArgParser::ArgParser(const int _argc, const char** _argv) : argc(_argc), argv(_argv)
{
}

const Args ArgParser::Parse()
{
	uint64_t num_points{0};
	uint32_t num_queries{0};
	uint8_t selectivity{0};
	Distribution point_distribution{Distribution::UNIFORM};
	float low{0};
	float high{0};

	static const std::string num_points_arg{"num_points"};
	static const std::string num_queries_arg{"num_queries"};
	static const std::string selectivity_arg{"selectivity"};
	static const std::string point_distribution_arg{"dist"};
	static const std::string value_range_low_arg{"low"};
	static const std::string value_range_high_arg{"high"};

	for(int32_t i = 1; i < this->argc; i++)
	{
		std::string arg{this->argv[i]};
		if(arg == num_points_arg)
		{
			num_points = stoll(this->argv[++i]);
		}
		else if(arg == num_queries_arg)
		{
			num_queries = stoi(this->argv[++i]);
		}
		else if(arg == selectivity_arg)
		{
			selectivity = stoi(this->argv[++i]);
		}
		else if(arg == point_distribution_arg)
		{
			std::string dist{this->argv[++i]};
			if(dist == "uniform")
			{
				point_distribution = Distribution::UNIFORM;
			}
			else if(dist == "gaussian")
			{
				point_distribution = Distribution::GAUSSIAN;
			}
			else
			{
				throw std::runtime_error("Unknown distribution " + dist);
			}
		}
		else if(arg == value_range_low_arg)
		{
			low = stof(this->argv[++i]);
		}
		else if(arg == value_range_high_arg)
		{
			high = stof(this->argv[++i]);
		}
		else
		{
			throw std::runtime_error("Unknown argument " + arg);
		}
	}

	return Args{num_points, num_queries, selectivity, point_distribution, ValueRange{low, high}};
}

}
}