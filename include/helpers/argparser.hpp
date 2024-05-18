//
// Created by aske on 4/11/24.
//

#ifndef GEO_RT_INDEX_ARGPARSER_HPP
#define GEO_RT_INDEX_ARGPARSER_HPP

#include "types/aabb.hpp"
#include "types/point_sorting.hpp"

#include <memory>
#include <string>
#include <string_view>
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
	types::PointSorting point_sort_type = types::PointSorting::None;
	std::string benchmark;
	int8_t lo;
	int8_t hi;
	bool compaction = false;
	std::string distribution;
	float aabb_z_value = 0;
	float ray_length = 1e9;
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
	inline types::PointSorting GetPointSorting() const
	{
		return this->point_sort_type;
	}
	inline const auto& GetLo() const
	{
		return this->lo;
	} 
	inline const auto& GetHi() const
	{
		return this->hi;
	} 
	inline const auto& GetBenchmark() const
	{
		return this->benchmark;
	} 
	inline const auto& GetDistribution() const
	{
		return this->distribution;
	}
	inline auto GetCompaction() const
	{
		return this->compaction;
	}
	inline auto GetAabbZValue() const
	{
		return this->aabb_z_value;
	}
	inline auto GetRayLength() const
	{
		return this->ray_length;
	}
};


}
}

#endif // GEO_RT_INDEX_ARGPARSER_HPP
