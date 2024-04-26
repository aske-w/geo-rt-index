//
// Created by aske on 4/14/24.
//

#ifndef GEO_RT_INDEX_AABB_HPP
#define GEO_RT_INDEX_AABB_HPP


namespace geo_rt_index
{
namespace types
{


//! Axis-aligned bounding box
struct Aabb
{
public:
	float minX, minY, maxX, maxY;
	Aabb(float _minX, float _minY, float _maxX, float _maxY) : minX(_minX), minY(_minY), maxX(_maxX), maxY(_maxY)
	{ }
	Aabb(int _minX, int _minY, int _maxX, int _maxY) : Aabb(static_cast<float>(_minX), static_cast<float>(_minY),
	           static_cast<float>(_maxX), static_cast<float>(_maxY))
	{ }
#ifdef __CUDACC__
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
	static inline const OptixAabb ToOptixAabb(const Aabb& aabb, float _minZ = 0, float _maxZ = 0)
	{
		return aabb.ToOptixAabb(_minZ, _maxZ);
	}
#endif
};

constexpr inline bool operator==(const Aabb& lhs, const Aabb& rhs)
{
	return lhs.minX == rhs.minX
		&& lhs.minY == rhs.minY
		&& lhs.maxX == rhs.maxX
		&& lhs.maxY == rhs.maxY;
};

}
}

#endif // GEO_RT_INDEX_AABB_HPP
