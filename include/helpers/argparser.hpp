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
private:
	bool debug;
	bool benchmark;
	std::vector<geo_rt_index::types::Aabb> queries;
	std::vector<std::string> files;
	inline static Args& GetMutableInstance()
	{
		static Args instance{};
		return instance;
	}
	explicit Args() { };
public:
	inline static const Args& GetInstance()
	{
		return GetMutableInstance();
	}
	static void Parse(const int, const char**);
public:
	Args(Args&) = delete;
	void operator=(const Args&) = delete;
	inline bool IsDebug() const
	{
		return this->debug;
	}
	inline bool IsBenchmark() const
	{
		return this->benchmark;
	}
	inline const std::vector<geo_rt_index::types::Aabb> GetQueries() const
	{
		return this->queries;
	}
	inline const std::vector<std::string> GetFiles() const
	{
		return this->files;
	}
};


}
}

#endif // GEO_RT_INDEX_ARGPARSER_HPP
