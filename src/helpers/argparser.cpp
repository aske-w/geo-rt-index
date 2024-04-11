//
// Created by aske on 4/11/24.
//

#include "helpers/argparser.hpp"
#include <exception>
#include <string>
#include <vector>

namespace geo_rt_index
{
namespace helpers
{

using std::stoi, std::stoll, std::stof;
using std::vector;
using std::string, std::string_view;

ValueRange::ValueRange(float _low, float _high) : low(_low), high(_high)
{
}

ArgParser::ArgParser(const int _argc, const char** _argv) : argc(_argc), argv(_argv)
{
}

static const bool IsCandidateArgument(const string_view& in_arg, const vector<string_view>& candidates)
{
	return std::find(candidates.cbegin(), candidates.cend(), in_arg) != candidates.end();
}

static const vector<string_view> num_points_args		{"-p", "--num_points"};
static const vector<string_view> num_queries_args		{"-q", "--num_queries"};
static const vector<string_view> selectivity_args		{"-s", "--selectivity"};
static const vector<string_view> point_distribution_args{"-d", "--dist"};
static const vector<string_view> value_range_low_args	{"-l", "--low"};
static const vector<string_view> value_range_high_args	{"-h", "--high"};

const Args ArgParser::Parse()
{
	uint64_t num_points{0};
	uint32_t num_queries{0};
	uint8_t selectivity{0};
	Distribution point_distribution{Distribution::UNIFORM};
	float low{0};
	float high{0};

	for(int32_t i = 1; i < this->argc; i++)
	{
		string arg{this->argv[i]};
		if(IsCandidateArgument(arg, num_points_args))
		{
			num_points = stoll(this->argv[++i]);
		}
		else if(IsCandidateArgument(arg, num_queries_args))
		{
			num_queries = stoi(this->argv[++i]);
		}
		else if(IsCandidateArgument(arg, selectivity_args))
		{
			selectivity = stoi(this->argv[++i]);
		}
		else if(IsCandidateArgument(arg, point_distribution_args))
		{
			string dist{this->argv[++i]};
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
		else if(IsCandidateArgument(arg, value_range_low_args))
		{
			low = stof(this->argv[++i]);
		}
		else if(IsCandidateArgument(arg, value_range_high_args))
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