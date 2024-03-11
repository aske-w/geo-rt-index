#ifndef CUDA_WRAPPER_HPP
#define CUDA_WRAPPER_HPP

#include <optix_types.h>
#include <bits/unique_ptr.h>
#include <optix_stubs.h>
#include "cuda_helpers.cuh"
#include "cuda_buffer.hpp"
#include "optix_helpers.cuh"

namespace geo_rt_index
{
namespace helpers
{

struct optix_wrapper {

    optix_wrapper(bool debug = false);
    ~optix_wrapper();

protected:
    void init_optix();

    void create_context();

    void create_module();

    void create_sphere_module();

public:
    CUcontext          cuda_context;
    cudaStream_t       stream;

    OptixDeviceContext optix_context;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions    pipeline_link_options = {};

    OptixModule                 module;
    OptixModule                 sphere_module;
    OptixModuleCompileOptions   module_compile_options = {};

    bool debug;
};

} // helpers
} // geo_rt_index



#endif // CUDA_WRAPPER_HPP