//
// Created by aske on 4/11/24.
//

#include "helpers/argparser.hpp"

#include "helpers/debug_helpers.hpp"

#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace geo_rt_index
{
namespace helpers
{

namespace fs = std::filesystem;

using std::stoi, std::stoll, std::stof;
using std::vector;
using std::string, std::string_view;


Args::Args(bool _debug, bool _benchmark, const vector<geo_rt_index::types::Aabb> _queries, const vector<std::string> _files)
    : debug(_debug), benchmark(false), queries(_queries), files(_files)
{
	if(debug)
	{
		D_PRINT("Debug enabled\n");
		D_PRINT("Number of queries: %zu\n",queries.size());
		D_PRINT("Number of files: %zu\n", files.size());
	}
}



ArgParser::ArgParser(const int _argc, const char** _argv) : argc(_argc), argv(_argv)
{
}

static const inline bool IsCandidateArgument(const string_view& in_arg, const vector<string_view>& candidates)
{
	return std::find(candidates.cbegin(), candidates.cend(), in_arg) != candidates.end();
}

static const vector<string_view> query_args{"-q", "--query"};
static const vector<string_view> debug_args{"-d", "--debug"};
static const vector<string_view> benchmark_args{"-b", "--benchmark"};

const Args ArgParser::Parse()
{
	bool debug{false};
	bool benchmark{false};
	vector<string> files;
	vector<types::Aabb> queries;

	for(int32_t i = 1; i < this->argc; i++)
	{
		const string arg{this->argv[i]};
		if(IsCandidateArgument(arg, query_args))
		{
			const float minx{stof(this->argv[++i])};
			const float miny{stof(this->argv[++i])};
			const float maxx{stof(this->argv[++i])};
			const float maxy{stof(this->argv[++i])};
			queries.emplace_back(minx, miny, maxx, maxy);
		}
		else if(fs::exists(arg))
		{
			files.push_back(arg);
		}
		else if(IsCandidateArgument(arg, debug_args))
		{
			debug = true;
		}
		else if(IsCandidateArgument(arg, benchmark_args))
		{
			benchmark = true;
		}
		else
		{
			throw std::runtime_error("Unknown argument: " + arg);
		}
	}
	return Args{debug, benchmark, queries, files};
}

}
}