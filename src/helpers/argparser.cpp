//
// Created by aske on 4/11/24.
//

#include "helpers/argparser.hpp"

#include "helpers/debug_helpers.hpp"

#include <exception>
#include <filesystem>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

namespace geo_rt_index
{
namespace helpers
{

namespace fs = std::filesystem;

using std::stoi, std::stoll, std::stof, std::stoull;
using std::vector;
using std::string, std::string_view;
using std::function;

static inline bool IsCandidateArgument(const string_view& in_arg, const vector<string_view>& candidates)
{
	return std::find(candidates.cbegin(), candidates.cend(), in_arg) != candidates.end();
}

static const vector<string_view> query_args{"-q", "--query"};
static const vector<string_view> aabb_layering_args{"-l", "--aabb-layering"};
static const vector<string_view> rays_per_thread_args{"-r", "--rays-per-thread"};
static const vector<string_view> num_repetitions_args{"-n", "--number-of-repetitions"};
static const vector<string_view> modifier_args{"-m", "--modifier"};
static const vector<string_view> id_args{"--id"};
static const vector<string_view> sort_args{"--sort"};
static const vector<string_view> benchmark_args{"--benchmark"};
static const vector<string_view> compact_flag_arg{"--compaction"};

void Args::Parse(const int argc, const char** argv)
{
	auto& instance = GetMutableInstance();
	for(int32_t i = 1; i < argc; i++)
	{
		const string arg{argv[i]};
		if(IsCandidateArgument(arg, query_args))
		{
			const float minx{stof(argv[++i])};
			const float miny{stof(argv[++i])};
			const float maxx{stof(argv[++i])};
			const float maxy{stof(argv[++i])};
			instance.queries.emplace_back(minx, miny, maxx, maxy);
		}
		else if(IsCandidateArgument(arg, aabb_layering_args))
		{
			const uint64_t input = stoull(argv[++i]);
			if(input > static_cast<uint64_t>(AabbLayering::Last))
			{
				throw std::runtime_error("AabbLayering input out of range, max is "
					+ std::to_string(static_cast<uint8_t>(AabbLayering::Last)));
			}
			instance.layering = static_cast<AabbLayering>(input);
		}
		else if(IsCandidateArgument(arg, rays_per_thread_args))
		{
			const auto input = stoi(argv[++i]);
			if (input < 0 || __builtin_clz(input) == 0)
			{
				throw std::runtime_error("u stoopid");
			}
			instance.rays_per_thread = static_cast<uint32_t>(input);
		}
		else if(IsCandidateArgument(arg, num_repetitions_args))
		{
			const auto input = stoi(argv[++i]);
			if (input < 0 || input > 255)
			{
				throw std::runtime_error("u stoopid");
			}
			instance.repetitions = static_cast<uint8_t>(input);
		}
		else if (IsCandidateArgument(arg, modifier_args))
		{
			const auto input = stof(argv[++i]);
			instance.modifier = input;
		}
		else if(IsCandidateArgument(arg, id_args))
		{
			instance.invocation_id = std::string{argv[++i]};
		}
		else if(IsCandidateArgument(arg, sort_args))
		{
			const uint64_t input = stoull(argv[++i]);
			if(input > static_cast<uint64_t>(types::PointSorting::Last))
			{
				throw std::runtime_error("types::PointSorting input out of range, max is "
				                         + std::to_string(static_cast<uint8_t>(types::PointSorting::Last)));
			}
			instance.point_sort_type = static_cast<types::PointSorting>(input);
		}
		else if(IsCandidateArgument(arg, benchmark_args))
		{
			auto b = std::string{argv[++i]};
			if(b.length() == 0)
			{
				throw std::runtime_error("benchmark arg may not be empty");
			}
			instance.benchmark = b;
		}
		else if(IsCandidateArgument(arg, compact_flag_arg))
		{
			instance.compaction = true;
		}
		else if(fs::exists(arg))
		{
			instance.files.push_back(arg);
		}
		else
		{
			if(arg.rfind('\\') != string::npos && fs::is_directory(arg.substr(0, arg.rfind('\\'))))
			{
				throw std::runtime_error("File does not exist: " + arg);
			}
			throw std::runtime_error("Unknown argument: " + arg);
		}
	}
	const auto mod = instance.GetModifier();
	for(auto& query : instance.queries)
	{
		query.minX *= mod;
		query.minY *= mod;
		query.maxX *= mod;
		query.maxY *= mod;
	}
	const auto& first_file = instance.files.at(0);
	if(first_file.find("_r01.parquet") != string::npos)
	{
		instance.lo = 0;
		instance.hi = 1;
	} 
	else if(first_file.find("_r-11.parquet") != string::npos)
	{
		instance.lo = -1;
		instance.hi = 1;
	}
	else
	{
		throw std::runtime_error("Could not determine dataset value range");
	}

	if(first_file.find("/uniform/") != string::npos)
	{
		instance.distribution = "uniform";
	}
	else if(first_file.find("/normal/") != string::npos)
	{
		instance.distribution = "normal";
	}
	else
	{
		throw std::runtime_error("Could not determine dataset distribution");
	}
}

}
}