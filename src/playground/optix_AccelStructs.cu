#include <optix_types.h>
#include <cstring>
#include "cuda_buffer.hpp"

/**
 * NVIDIA OptiX provides acceleration structures to optimize the search for the intersection of rays with the geometric
 * data in the scene
 */
using std::vector;

int main(){
    OptixAccelBuildOptions options = {};
    const constexpr uint8_t numInputs = 1;
    OptixBuildInput inputs[numInputs];
//    CUdeviceptr tempBuffer, outputBuffer;
//    size_t tempBufferSize, outputBufferSize;
    memset(&options, 0, sizeof(OptixAccelBuildOptions));
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    options.motionOptions.numKeys = 0;
    memset(inputs, 0, sizeof(OptixBuildInput) * numInputs);


    cuda_buffer buf;
    buf.alloc_and_upload(vector<float3>{
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
    });

    OptixBuildInputTriangleArray& triangleInput =
            inputs[0].triangleArray;
//    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    triangleInput.vertexBuffers = (CUdeviceptr*)buf.raw_ptr;
    triangleInput.numVertices = numInputs * 3;
    triangleInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.vertexStrideInBytes = sizeof( float3 );
    triangleInput.indexBuffer = 0;
    triangleInput.numIndexTriplets = 0;
    triangleInput.indexFormat = OPTIX_INDICES_FORMAT_NONE;
//    triangleInput.indexStrideInBytes = sizeof( float3 );
    triangleInput.preTransform = 0;
    triangleInput.numSbtRecords = 1;
}