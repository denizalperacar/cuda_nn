#ifndef CUDA_NN_LIB_SRC_VECTOR_VEC_H_
#define CUDA_NN_LIB_SRC_VECTOR_VEC_H_

#include "namespaces.h"
#include "helpers.h"

#include <cstdint>

CNNL_NAMESPACE_BEGIN

template <typename T, uint32_t DIM>
struct VECTOR_ATTR vec {
	vec() = default;

	CNNL_HOST_DEVICE vec(T initialize_to) {
		CNNL_PRAGMA_UNROLL
		for (uint32_t i = 0; i < DIM; i++) {
			data[i] = initialize_to;
		}
	}

	template <typename U, uint32_t D>
	CNNL_HOST_DEVICE vec(const vec<U, D>& obj) {
		CNNL_PRAGMA_UNROLL
		for (uint32_t i = 0; i < min(D, DIM); i++) {
			data[i] = (T)obj[i];
		}
	}

	CNNL_HOST_DEVICE T& operator[](uint32_t index) {
		return data[index];
	}

	CNNL_HOST_DEVICE T operator[](uint32_t index) const {
		return data[index];
	}

	CNNL_HOST_DEVICE T& at(uint32_t index) {
		return data[index];
	}

	CNNL_HOST_DEVICE T at(uint32_t index) const {
		return data[index];
	}

	CNNL_HOST_DEVICE static constexpr uint32_t size() {
		return N;
	}

	T data[DIM];
	static constexpr uint32_t N = DIM;
};




CNNL_NAMESPACE_END

#endif
