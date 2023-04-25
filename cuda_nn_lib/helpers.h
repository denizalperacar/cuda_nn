#ifndef CUDA_NN_LIB_SRC_COMMON_HELPERS_H_
#define CUDA_NN_LIB_SRC_COMMON_HELPERS_H_

#include "namespaces.h"

CNNL_NAMESPACE_BEGIN


// paradigm helpers
#if defined(__NVCC__) || defined(__clang__) && defined(__CUDA__)
#define CNNL_HOST_DEVICE __host__ __device__
#define CNNL_DEVICE __device__
#define CNNL_HOST __host__
#define CNNL_DEVICE_GLOBAL __global__
#define CNNL_SHARED_MEM __shared__
#else 
#define CNNL_HOST_DEVICE 
#define CNNL_DEVICE 
#define CNNL_HOST 
#define CNNL_DEVICE_GLOBAL 
#define CNNL_SHARED_MEM 
#endif

#ifndef VECTOR_ATTR
#define VECTOR_ATTR
#endif

#if defined(__CUDA__ARCH__)

	#define CNNL_PRAGMA_UNROLL #pragma unroll
	#define CNNL_PRAGMA_NO_UNROLL #pragma unroll 1

#else 

	#define CNNL_PRAGMA_UNROLL 
	#define CNNL_PRAGMA_NO_UNROLL 

#endif
CNNL_NAMESPACE_END

#endif