#ifndef GEO_RT_INDEX_SPATIAL_HELPERS_CUH
#define GEO_RT_INDEX_SPATIAL_HELPERS_CUH

#include "types.hpp"

namespace geo_rt_index
{
namespace helpers
{

class SpatialHelpers
{
public:
	//! Should match https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/#spatial-filtering-functions
	static __host__ __device__ __forceinline__ bool Contains(const Aabb& aabb, const Point& point)
	{
		return aabb.minX < point.x && point.x < aabb.maxX &&
			   aabb.minY < point.y && point.y < aabb.maxY;
	}
};

} // helpers
} // geo_rt_index

#endif // GEO_RT_INDEX_SPATIAL_HELPERS_CUH