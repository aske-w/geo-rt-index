#include <bits/unique_ptr.h>
#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "cuda_helpers.cuh"
#include "optix_helpers.cuh"

using std::cout;

#if DEBUG
const constexpr int log_level = 4;
const constexpr OptixDeviceContextValidationMode validation_mode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
const constexpr OptixDeviceContextValidationMode validation_mode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
const constexpr int log_level = 1;
#endif


void cb(unsigned int level,
        const char* tag,
        const char* message,
        void* cbdata) {
    cout << "LEVEL: " << level << '\n'
    << "TAG: " << tag << '\n'
    << "MSG: " << message << '\n';
}

int main() {
    cudaFree(0); CUERR
    OptixDeviceContext context = nullptr;
    CUcontext cuCtx = 0;
    const OptixDeviceContextOptions options = {
        .logCallbackFunction = cb,
        .logCallbackLevel = log_level,
        .validationMode = validation_mode
    };
    OPTIX_CHECK(optixInit())
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context))

    return 0;
}