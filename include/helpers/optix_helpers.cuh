#ifndef OPTIX_HELPERS_CUH
#define OPTIX_HELPERS_CUH

#include <optix.h>

namespace geo_rt_index
{
namespace helpers
{

#ifdef __CUDACC__
#define DEVICEQUALIFIER  __device__
#else
#define DEVICEQUALIFIER
#endif


#ifdef __CUDACC__
#define INLINEQUALIFIER  __forceinline__
#else
#define INLINEQUALIFIER inline
#endif

#ifdef DEBUG
#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }
#else
#define OPTIX_CHECK(call) call;
#endif


template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
    void set_payload_32(packed_type i) {
	static_assert(sizeof(packed_type) == 4);
	optixSetPayload_0(i);
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
    packed_type get_payload_32() {
	static_assert(sizeof(packed_type) == 4);
	return (packed_type) optixGetPayload_0();
}

} // helpers
} // geo_rt_index

#endif // OPTIX_HELPERS_CUH
