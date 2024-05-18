#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "launch_parameters.hpp"

#include <cuda_runtime.h>

__global__ void KernelFilter(const LaunchParameters* const params);

#endif