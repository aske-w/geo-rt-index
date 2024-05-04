#include "helpers/debug_helpers.hpp"
#include "helpers/optix_wrapper.hpp"
#include "helpers/time.hpp"
#include "optix_stubs.h"

#include <optix_stubs.h>

extern "C" char embedded_ptx_code[];

using namespace geo_rt_index::helpers;

optix_wrapper::optix_wrapper(bool debug) : debug{debug} {
	MEASURE_TIME("optix_wrapper ctor",
		init_optix();
		create_context();
		create_module();
	#if PRIMITIVE == 1
		create_sphere_module();
	#endif
	);
}

optix_wrapper::~optix_wrapper() {
    OPTIX_CHECK(optixModuleDestroy(module));
#if PRIMITIVE == 1
    OPTIX_CHECK(optixModuleDestroy(sphere_module));
#endif
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context));
    cudaStreamDestroy(stream); CUERR
}


/*! helper function that initializes optix and checks for errors */
void optix_wrapper::init_optix() {
    cudaFree(0);
    int num;
    cudaGetDeviceCount(&num);
    if (num == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    OPTIX_CHECK(optixInit());
}


static void noop_context_log_cb(unsigned int, const char*, const char*, void*)
{
	return;
}

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *) {
    // ENABLE IF NEEDED
	D_PRINT("[%2d][%12s]: %s\n", (int)level, tag, message);
}

static OptixLogCallback get_context_log_cb(bool debug)
{
	if(debug)
	{
		return context_log_cb;
	}
	return noop_context_log_cb;
}

static void print_log(const char *message) {
    // ENABLE IF NEEDED
    // std::cout << "log=" << message << std::endl;
}

#if DEBUG
const constexpr int log_level = 4;
const constexpr OptixDeviceContextValidationMode validation_mode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
const constexpr OptixDeviceContextValidationMode validation_mode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
const constexpr int log_level = 1;
#endif


void optix_wrapper::create_context() {
    cudaSetDevice(0); CUERR
    cudaStreamCreate(&stream); CUERR
    cuCtxGetCurrent(&cuda_context); CUERR
	const OptixDeviceContextOptions options = {
	    .logCallbackFunction = get_context_log_cb(this->debug),
	    .logCallbackLevel = log_level,
	    .validationMode = validation_mode
	};
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &options, &optix_context));
//    OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context, context_log_cb, nullptr, 4));
}


void optix_wrapper::create_module() {

    // figure out payload semantics and register usage impact
    // https://raytracing-docs.nvidia.com/optix7/guide/index.html#payload

    module_compile_options.maxRegisterCount  = 0;
    module_compile_options.optLevel          = debug ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 : OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = debug ? OPTIX_COMPILE_DEBUG_LEVEL_FULL : OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    // set this for profiling
    //module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.numPayloadValues      = 2;
    pipeline_compile_options.numAttributeValues    = 0;
    pipeline_compile_options.exceptionFlags        = debug ? (OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH) : OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
#if PRIMITIVE == 1
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#endif

    pipeline_link_options.maxTraceDepth = 2;

    const std::string ptx { embedded_ptx_code };

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
            optix_context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx.c_str(),
            ptx.length(),
            log,&sizeof_log,
            &module
    ));
    if (sizeof_log > 1) print_log(log);
}


void optix_wrapper::create_sphere_module() {
    OptixBuiltinISOptions builtin_is_options = {};
    builtin_is_options.usesMotionBlur      = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK(optixBuiltinISModuleGet(optix_context, &module_compile_options, &pipeline_compile_options, &builtin_is_options, &sphere_module));
}
