#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <optix_types.h>

struct triangle {
	float3 v1, v2, v3; // TODO const?
	constexpr inline static const uint32_t vertex_count() {
		return 3;
	};
};

#endif // TYPES_HPP