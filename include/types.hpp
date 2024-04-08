#ifndef TYPES_HPP
#define TYPES_HPP

#include "helpers/cuda_buffer.hpp"
#include "helpers/debug_helpers.hpp"
#include "helpers/exception.hpp"
#include "helpers/general.hpp"

#include <cstdint>
#include <cstdio>
#include <optix_types.h>

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
};

struct Point {
	float x, y;

	Point(int _x, int _y) : x(static_cast<float>(_x)), y(static_cast<float>(_y))
	{
	}
	Point(float _x, float _y) : x(_x), y(_y)
	{
	}

	Triangle ToTriangle() const {
		const constexpr float f = 3.f;
		auto t = Triangle{
		    {x, 0 - (0.5f * f), -1},
		    {x, 0 + (0.5f * f), -1},
		    {x, 0, 1}
		};
		D_PRINT("((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))",
		       t.v1.x, t.v1.y, t.v1.z,
		       t.v2.x, t.v2.y, t.v2.z,
		       t.v3.x, t.v3.y, t.v3.z);
		return t;
	}
};

//! Axis-aligned bounding box
struct Aabb
{
public:
	float minX, minY, maxX, maxY;
	Aabb(float _minX, float _minY, float _maxX, float _maxY) : minX(_minX), minY(_minY), maxX(_maxX), maxY(_maxY)
	{
		if(_minX >= _maxX)
		{
			throw ArgumentException{nameof(_maxX), string_format("Must be greater than %s", nameof(_minX))};
		}
		if(_minY >= _maxY)
		{
			throw ArgumentException{nameof(_maxY), string_format("Must be greater than %s", nameof(_minY))};
		}
	}
	Aabb(int _minX, int _minY, int _maxX, int _maxY) : Aabb(static_cast<float>(_minX), static_cast<float>(_minY),
	           static_cast<float>(_maxX), static_cast<float>(_maxY))
	{ }
	inline constexpr const OptixAabb ToOptixAabb(float _minZ = 0, float _maxZ = 0) const
	{
		return std::move(OptixAabb
		{
			.minX = this->minX,
			.minY = this->minY,
			.minZ = _minZ,
			.maxX = this->maxX,
			.maxY = this->maxY,
			.maxZ = _maxZ
		});
	}
};

constexpr inline bool operator==(const Aabb& lhs, const Aabb& rhs)
{
	return lhs.minX == rhs.minX
		&& lhs.minY == rhs.minY
		&& lhs.maxX == rhs.maxX
		&& lhs.maxY == rhs.maxY;
};

enum class IndexType : uint8_t
{
	RESERVED = 0,
	PTA = 1
};


} // types
} // geo_rt_index

#endif // TYPES_HPP