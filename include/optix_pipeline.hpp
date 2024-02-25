#ifndef OPTIX_PIPELINE_HPP
#define OPTIX_PIPELINE_HPP

#include "optix_wrapper.hpp"

struct optix_pipeline {
    optix_pipeline(optix_wrapper* optix, bool verbose = false);
    ~optix_pipeline();

protected:
    void create_raygen_programs();

    void create_miss_programs();

    void create_hitgroup_programs();

    void assemble_pipeline();

    void build_sbt();

    bool verbose;

public:
    optix_wrapper* optix;
    OptixPipeline pipeline;

    std::vector<OptixProgramGroup> raygen_program_groups;
    cuda_buffer raygen_records_buffer;
    std::vector<OptixProgramGroup> miss_program_groups;
    cuda_buffer miss_records_buffer;
    std::vector<OptixProgramGroup> hitgroup_program_groups;
    cuda_buffer hitgroup_records_buffer;
    OptixShaderBindingTable sbt = {};
};

#endif // OPTIX_PIPELINE_HPP