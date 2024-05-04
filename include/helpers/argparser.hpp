//
// Created by aske on 4/11/24.
//

#ifndef GEO_RT_INDEX_ARGPARSER_HPP
#define GEO_RT_INDEX_ARGPARSER_HPP

#include "types/aabb.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <uuid/uuid.h>
#include <vector>

/**

/home/aske/dev/geo-rt-index/data/duniform_p22_s1337.parquet
/home/aske/dev/geo-rt-index/data/duniform_p26_s10422.parquet

 */
namespace geo_rt_index
{
namespace helpers
{

enum class AabbLayering : uint8_t
{
	None,
	Stacked,
	StackedSpaced,

	First = None,
	Last = StackedSpaced
};

class Args
{
private:
	std::vector<geo_rt_index::types::Aabb> queries;
	std::vector<std::string> files;
	AabbLayering layering = AabbLayering::None;
	uint32_t rays_per_thread = 1;
	uint8_t repetitions;
	float modifier = 1;
	std::string invocation_id;
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
	inline const std::vector<geo_rt_index::types::Aabb>& GetQueries() const
	{
		return this->queries;
	}
	inline const std::vector<std::string>& GetFiles() const
	{
		return this->files;
	}
	inline AabbLayering GetLayering() const
	{
		return this->layering;
	}
	inline auto GetRaysPerThread() const
	{
		return rays_per_thread;
	}
	inline auto GetRepetitions() const
	{
		return this->repetitions;
	}
	inline auto GetModifier() const
	{
		return this->modifier;
	}
	inline const std::string& GetID() const
	{
		return this->invocation_id;
	}
};


}
}

#endif // GEO_RT_INDEX_ARGPARSER_HPP
