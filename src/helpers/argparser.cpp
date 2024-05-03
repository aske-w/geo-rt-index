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
}

}
}