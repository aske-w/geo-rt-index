/**
 * This file is adapted from the RTIndeX equivalent
 */
#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH
#include <iostream>

namespace geo_rt_index
{
namespace helpers
{

#ifdef __CUDACC__
#ifdef DEBUG
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << '\n';       \
            exit(1);                                                           \
        }                                                                      \
    }
#else
#define CUERR

#endif
#endif

} // helpers
} // geo_rt_index

#endif // CUDA_HELPERS_CUH