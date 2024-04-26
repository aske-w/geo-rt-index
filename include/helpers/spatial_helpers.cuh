#ifndef GEO_RT_INDEX_SPATIAL_HELPERS_CUH
#define GEO_RT_INDEX_SPATIAL_HELPERS_CUH

#include "types.hpp"

namespace geo_rt_index
{
namespace helpers
{

using geo_rt_index::types::Point;
using geo_rt_index::types::Aabb;

class SpatialHelpers
{
public:
	//! Should match https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/#spatial-filtering-functions
	static __host__ __device__ __forceinline__ bool Contains(const types::Aabb& aabb, const types::Point& point)
	{
		return aabb.minX < point.x && point.x < aabb.maxX &&
			   aabb.minY < point.y && point.y < aabb.maxY;
	}
	//! Should match https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/#spatial-filtering-functions
	static __host__ __device__ __forceinline__ bool Contains(const OptixAabb& aabb, const types::Point& point)
	{
		return aabb.minX < point.x && point.x < aabb.maxX &&
		       aabb.minY < point.y && point.y < aabb.maxY;
	}
};

} // helpers
} // geo_rt_index

#endif // GEO_RT_INDEX_SPATIAL_HELPERS_CUH