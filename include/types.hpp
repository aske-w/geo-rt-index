#ifndef TYPES_HPP
#define TYPES_HPP

#include "helpers/cuda_buffer.hpp"
#include "helpers/debug_helpers.hpp"
#include "helpers/exception.hpp"
#include "helpers/general.hpp"

#include <cstdint>
#include <cstdio>
#include <optix_types.h>
#include <vector_types.h>
#include "types/point.hpp"
#include "types/aabb.hpp"


namespace geo_rt_index
{
namespace types
{


using geo_rt_index::helpers::string_format;
using geo_rt_index::helpers::ArgumentException;

struct Triangle
{
	float3 v1, v2, v3; // TODO const?
	Triangle(float3 _v1, float3 _v2, float3 _v3): v1(_v1), v2(_v2), v3(_v3)
	{
	}

	constexpr inline static uint32_t vertex_count()
	{
		return 3;
	};

	constexpr inline static uint32_t vertex_bytes()
	{
		return sizeof(float3);
	}
	static Triangle FromPoint(const geo_rt_index::types::Point& p) {
		const constexpr float f = 3.f;
		auto t = Triangle{
		    {p.x, 0 - (0.5f * f), -1},
		    {p.x, 0 + (0.5f * f), -1},
		    {p.x, 0, 1}
		};
		D_PRINT("((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))",
		        t.v1.x, t.v1.y, t.v1.z,
		        t.v2.x, t.v2.y, t.v2.z,
		        t.v3.x, t.v3.y, t.v3.z);
		return t;
	}
};

enum class IndexType : uint8_t
{
	RESERVED = 0,
	PTA = 1
};


} // types
} // geo_rt_index

#endif // TYPES_HPP