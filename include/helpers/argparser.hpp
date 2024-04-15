//
// Created by aske on 4/11/24.
//

#ifndef GEO_RT_INDEX_ARGPARSER_HPP
#define GEO_RT_INDEX_ARGPARSER_HPP

#include "types/aabb.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

/**

/home/aske/dev/geo-rt-index/data/duniform_p22_s1337.parquet /home/aske/dev/geo-rt-index/data/duniform_p26_s10422.parquet

 */
namespace geo_rt_index
{
namespace helpers
{

class Args
{
public:
	const bool debug;
	const bool benchmark;
	const std::vector<geo_rt_index::types::Aabb> queries;
	const std::vector<std::string> files;
	explicit Args(bool, bool, const std::vector<geo_rt_index::types::Aabb>, const std::vector<std::string>);
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
