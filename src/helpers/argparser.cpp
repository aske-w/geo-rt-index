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


//Args::Args(bool _debug, bool _benchmark, const vector<geo_rt_index::types::Aabb> _queries, const vector<std::string> _files)
//    : debug(_debug), benchmark(false), queries(_queries), files(_files)
//{
//	if(debug)
//	{
//		D_PRINT("Debug enabled\n");
//		D_PRINT("Number of queries: %zu\n",queries.size());
//		D_PRINT("Number of files: %zu\n", files.size());
//	}
//}


static inline bool IsCandidateArgument(const string_view& in_arg, const vector<string_view>& candidates)
{
	return std::find(candidates.cbegin(), candidates.cend(), in_arg) != candidates.end();
}

static const vector<string_view> query_args{"-q", "--query"};
static const vector<string_view> debug_args{"-d", "--debug"};
static const vector<string_view> benchmark_args{"-b", "--benchmark"};

void Args::Parse(const int argc, const char** argv)
{
//	bool debug{false};
//	bool benchmark{false};
//	vector<string> files;
//	vector<types::Aabb> queries;

	for(int32_t i = 1; i < argc; i++)
	{
		const string arg{argv[i]};
		if(IsCandidateArgument(arg, query_args))
		{
			const float minx{stof(argv[++i])};
			const float miny{stof(argv[++i])};
			const float maxx{stof(argv[++i])};
			const float maxy{stof(argv[++i])};
			GetMutableInstance().queries.emplace_back(minx, miny, maxx, maxy);
		}
		else if(fs::exists(arg))
		{
			GetMutableInstance().files.push_back(arg);
		}
		else if(IsCandidateArgument(arg, debug_args))
		{
			GetMutableInstance().debug = true;
		}
		else if(IsCandidateArgument(arg, benchmark_args))
		{
			GetMutableInstance().benchmark = true;
		}
		else
		{
			throw std::runtime_error("Unknown argument: " + arg);
		}
	}
}

}
}