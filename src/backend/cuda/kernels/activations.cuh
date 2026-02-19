/*
 * activations.cuh
 *
 *  Created on: Feb 17, 2026
 *      Author: maciek
 */

#ifndef BACKEND_CUDA_KERNELS_ACTIVATIONS_CUH_
#define BACKEND_CUDA_KERNELS_ACTIVATIONS_CUH_

#include <minml/backend/backend_utils.hpp>

#include "../helpers/misc.cuh"
#include "../vec/vec_headers.cuh"

template<typename T, int N>
__device__ vectors::vec<T, N> activation_forward(ml::mlActivationType_t act, const vectors::vec<T, N> &x)
{
	switch (act)
	{
		default:
		case ml::ACTIVATION_LINEAR:
			return x;
		case ml::ACTIVATION_SIGMOID:
			return vectors::sigmoid(x);
		case ml::ACTIVATION_TANH:
			return vectors::tanh(x);
		case ml::ACTIVATION_RELU:
			return vectors::relu(x);
		case ml::ACTIVATION_LEAKY_RELU:
			return select(x > vectors::zero<T, N>(), x, x * vectors::vec<T, N>(0.1f));
		case ml::ACTIVATION_EXP:
			return vectors::exp(x);
	}
}

template<typename T, int N>
__device__ vectors::vec<T, N> activation_backward(ml::mlActivationType_t act, const vectors::vec<T, N> &gradient, const vectors::vec<T, N> &input,
		const vectors::vec<T, N> &output)
{
	switch (act)
	{
		default:
		case ml::ACTIVATION_LINEAR:
			return gradient;
		case ml::ACTIVATION_SIGMOID:
			return gradient * output * (vectors::one<T, N>() - output);
		case ml::ACTIVATION_TANH:
			return gradient * (vectors::one<T, N>() - square(output));
		case ml::ACTIVATION_RELU:
			return select(output > vectors::zero<T, N>(), gradient, vectors::zero<T, N>());
		case ml::ACTIVATION_LEAKY_RELU:
			return select(output > vectors::zero<T, N>(), gradient, gradient * vectors::vec<T, N>(0.1f));
		case ml::ACTIVATION_EXP:
			return gradient * output;
	}
}

#endif /* BACKEND_CUDA_KERNELS_ACTIVATIONS_CUH_ */
