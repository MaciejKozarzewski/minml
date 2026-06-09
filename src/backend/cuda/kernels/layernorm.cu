/*
 * layernorm.cu
 *
 *  Created on: Mar 28, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec_headers.cuh"
#include "../helpers/misc.cuh"
#include "../helpers/indexers.cuh"
#include "../helpers/AvgVarStats.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;

	template<typename T, int N>
	__device__ AvgVarStats<T> get_stats(const vec<T, N> &v)
	{
		const T mean = horizontal_add(v) / static_cast<T>(N);
		const T var = horizontal_add(square(v - mean));
		return AvgVarStats<T>(static_cast<T>(v.size()), mean, var);
	}
	template<typename T>
	__device__ AvgVarStats<T> get_stats(const vec<T, 1> &v)
	{
		AvgVarStats<T> result;
		result.add(v.x0);
		return result;
	}

	template<typename T>
	__device__ T get_inv_stddev(T variance, int N, float epsilon);

	template<>
	__device__ float get_inv_stddev(float variance, int N, float epsilon)
	{
		return 1.0f / std::sqrt(epsilon + variance / (N - 1));
	}

	template<typename T, int N, typename U = T>
	__global__ void kernel_layernorm_forward(const T *input, T *output, const T *weights, const T *bias, const T *ext, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 32);
		assert(blockDim.y == 8);

		extern __shared__ char shared_array[];

		T *shared_input = reinterpret_cast<T*>(shared_array);
		T *shared_weights = shared_input + last_dim;
		T *shared_bias = shared_weights + last_dim;

		const int tid = blockIdx.y * blockDim.x + threadIdx.x;

		for (int j = N * tid; j < last_dim; j += N * blockDim.x * blockDim.y)
		{
			if (weights != nullptr)
				vector_copy<N>(shared_weights + j, weights + j);
			else
				store_vec(shared_weights + j, one<T, N>());
			if (bias != nullptr)
				vector_copy<N>(shared_bias + j, bias + j);
			else
				store_vec(shared_bias + j, zero<T, N>());
		}
		__syncthreads();

		for (int i = blockIdx.x * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.x * blockDim.y)
		{
			U avg = static_cast<U>(0.0f);
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> in(input + i * last_dim + j);
				avg += (horizontal_add(in));
				store_vec(shared_input + j, in);
			}
			for (int k = 16; k >= 1; k /= 2)
				avg += __shfl_xor_sync(0xffffffff, avg, k);
			avg /= static_cast<U>(last_dim);

			const vec<T, N> vec_avg(avg);
			U var = static_cast<U>(0.0f);
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> in(shared_input + j);
				var += static_cast<U>(horizontal_add(square(in - vec_avg)));
			}
			for (int k = 16; k >= 1; k /= 2)
				var += __shfl_xor_sync(0xffffffff, var, k);

			const vec<T, N> inv_stddev(static_cast<U>(1.0f) / std::sqrt(static_cast<U>(1.0e-6f) + var / (last_dim - 1)));

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> gamma(shared_weights + j);
				const vec<T, N> beta(shared_bias + j);
				const vec<T, N> in(shared_input + j);
				const vec<T, N> out = gamma * (in - avg) * inv_stddev + beta;
				out.store(output + i * last_dim + j);
			}
		}
	}
	template<typename T, int N, typename U = T>
	__global__ void kernel_layernorm_backward(const T *input, float beta_prev, T *gradient_prev, T *gradient_next, const T *weights, float beta_w,
			T *weights_update, float beta_b, T *bias_update, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);

		extern __shared__ char shared_array[];

		T *shared_input = reinterpret_cast<T*>(shared_array);
		T *shared_gradient = shared_input + last_dim;
		T *shared_weights = shared_gradient + last_dim;
		T *shared_weights_update = shared_weights + last_dim;
		T *shared_bias_update = shared_weights_update + last_dim;

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		vec<float, N> thread_weights_update(0.0f);
		vec<float, N> thread_bias_update(0.0f);

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			const vec<float, N> zero(0.0f);
			zero.store(shared_weights_update + j);
			zero.store(shared_bias_update + j);
			vector_copy<N>(shared_weights + j, weights + j);
		}

		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(input + i * last_dim + j);
				const vec<float, N> grad = load_vec<float, N>(gradient_next + i * last_dim + j);
				avg += horizontal_add(in);
				store_vec(shared_input + j, in);
				store_vec(shared_gradient + j, grad);
			}
			avg = cg::reduce(tile, avg, cg::plus<float>()) / last_dim;

			float var = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
				var += square(shared_input[j] - avg);
			const float inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<float>()), last_dim, 1.0e-6f);

			float d_sigma = 0.0f;
			float d_mu = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const float in = (shared_input[j] - avg) * inv_stddev;
				const float grad = shared_gradient[j];
				const float gamma = shared_weights[j];

				d_sigma -= grad * in * gamma;
				d_mu -= grad * gamma;
				shared_weights_update[j] += grad * in;
				shared_bias_update[j] += grad;

				shared_input[j] = in;
				shared_gradient[j] = grad * gamma;
			}

			d_sigma = cg::reduce(tile, d_sigma, cg::plus<float>()) * inv_stddev / (last_dim - 1);
			d_mu = cg::reduce(tile, d_mu, cg::plus<float>()) * inv_stddev / last_dim;

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				const vec<float, N> grad = load_vec<float, N>(shared_gradient + j);
				const vec<float, N> tmp = grad * inv_stddev + d_sigma * in + d_mu;
				store_vec(gradient_prev + i * last_dim + j, tmp);
			}
		}
		__syncthreads();
		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			vector_copy<N>(weights_update + blockIdx.x * last_dim + j, shared_weights_update + j);
			vector_copy<N>(bias_update + blockIdx.x * last_dim + j, shared_bias_update + j);
		}
	}

	template<typename T, int N>
	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward_v2(const T *input, T *output, const T *weights, const T *bias, const T *ext, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 256);

		extern __shared__ char shared_array[];

		float *shared_input = reinterpret_cast<float*>(shared_array);
		float *shared_weights = shared_input + last_dim;
		float *shared_bias = shared_weights + last_dim;

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			if (weights != nullptr)
				vector_copy<N>(shared_weights + j, weights + j);
			else
				store_vec(shared_weights + j, one<T, N>());
			if (bias != nullptr)
				vector_copy<N>(shared_bias + j, bias + j);
			else
				store_vec(shared_bias + j, zero<T, N>());
		}
		__syncthreads();

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			float avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(input + i * last_dim + j);
				avg += horizontal_add(in);
				store_vec(shared_input + j, in);
			}
			avg = cg::reduce(tile, avg, cg::plus<float>()) / static_cast<float>(last_dim);

			float var = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				var += horizontal_add(square(in - avg));
			}
			const float inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<float>()), last_dim, 1.0e-6f);

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<float, N> gamma = load_vec<float, N>(shared_weights + j);
				const vec<float, N> beta = load_vec<float, N>(shared_bias + j);
				const vec<float, N> in = load_vec<float, N>(shared_input + j);
				const vec<float, N> out = gamma * (in - avg) * inv_stddev + beta;
				store_vec(output + i * last_dim + j, out);
			}
		}
	}
	template<typename T, int N>
	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward_v3(const T *input, T *output, const T *weights, const T *bias, const T *ext, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 256);

		extern __shared__ char shared_array[];

		T *shared_weights = reinterpret_cast<T*>(shared_array);
		T *shared_bias = shared_weights + last_dim;

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			const vec<T, N> w = (weights != nullptr) ? load_vec<T, N>(weights + j) : one<T, N>();
			const vec<T, N> b = (bias != nullptr) ? load_vec<T, N>(bias + j) : zero<T, N>();
			store_vec(shared_weights + j, w);
			store_vec(shared_bias + j, b);
		}
		__syncthreads();

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			vec<T, N> in;

			T avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				in = load_vec<T, N>(input + i * last_dim + j);
				avg += horizontal_add(in);
			}
			avg = cg::reduce(tile, avg, cg::plus<T>()) / static_cast<T>(last_dim);

			T var = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				in -= avg;
				var += horizontal_add(square(in));
			}
			const T inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<T>()), last_dim, 1.0e-6f);

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> gamma = load_vec<T, N>(shared_weights + j);
				const vec<T, N> beta = load_vec<T, N>(shared_bias + j);
				const vec<T, N> out = gamma * in * inv_stddev + beta;
				store_vec(output + i * last_dim + j, out);
			}
		}
	}
	template<typename T, int N>
	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward_vect_v4(const T *input, T *output, const T *weights, const T *bias, int first_dim, int last_dim)
	{
		assert(blockDim.x * N == last_dim);

		extern __shared__ char shared_array[];

		T *shared_weights = reinterpret_cast<T*>(shared_array);
		T *shared_bias = shared_weights + last_dim;

		if (threadIdx.y == 0)
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<T, N> w = load_vec<T, N>(weights + j);
				const vec<T, N> b = load_vec<T, N>(bias + j);
				store_vec(shared_weights + j, w);
				store_vec(shared_bias + j, b);
			}
		__syncthreads();

		const int tid = N * threadIdx.x;

		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
		{
			vec<T, N> in = (tid < last_dim) ? vec<T, N>(input + i * last_dim + tid) : zero<T, N>();

			float avg = horizontal_add(in);
			for (int k = 16; k >= 1; k /= 2)
				avg += __shfl_xor_sync(0xffffffff, avg, k);
			avg /= last_dim;

			in -= vec<T, N>(avg);
			float var = horizontal_add(square(in));
			for (int k = 16; k >= 1; k /= 2)
				var += __shfl_xor_sync(0xffffffff, var, k);
			const T inv_stddev = get_inv_stddev(var, last_dim, 1.0e-6f);

			if (tid < last_dim)
			{
				const vec<T, N> gamma(shared_weights + tid);
				const vec<T, N> beta(shared_bias + tid);
				const vec<T, N> out = gamma * in * inv_stddev + beta;
				out.store(output + i * last_dim + tid);
			}
		}
	}
	template<typename T, int N, typename U = T>
	__launch_bounds__(256, 4)
	__global__ void kernel_layernorm_forward_v4(const T *input, T *output, const T *weights, const T *bias, int first_dim, int last_dim, float alpha,
			float beta_y, ml::mlActivationType_t act)
	{
		assert(last_dim % N == 0);
		assert(blockDim.x == 32);

		extern __shared__ char shared_array[];

		U *shared_input = reinterpret_cast<U*>(shared_array);
		U *shared_weights = shared_input + blockDim.y * last_dim;
		U *shared_bias = shared_weights + last_dim;

		if (threadIdx.y == 0)
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				if (weights != nullptr)
					vector_copy<N>(shared_weights + j, weights + j);
				else
					store_vec(shared_weights + j, one<T, N>());
				if (bias != nullptr)
					vector_copy<N>(shared_bias + j, bias + j);
				else
					store_vec(shared_bias + j, zero<T, N>());
			}
		__syncthreads();

		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
		{
			U avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<U, N> in = load_vec<U, N>(input + i * last_dim + j);
				avg += horizontal_add(in);
				store_vec(shared_input + threadIdx.y * last_dim + j, in);
			}
			for (int k = 16; k >= 1; k /= 2)
				avg += __shfl_xor_sync(0xffffffff, avg, k);
			avg /= static_cast<U>(last_dim);

			U var = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<U, N> in = load_vec<U, N>(shared_input + threadIdx.y * last_dim + j);
				var += horizontal_add(square(in - avg));
			}
			for (int k = 16; k >= 1; k /= 2)
				var += __shfl_xor_sync(0xffffffff, var, k);
			const U inv_stddev = get_inv_stddev(var, last_dim, 1.0e-6f);

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<U, N> gamma(shared_weights + j);
				const vec<U, N> beta(shared_bias + j);
				const vec<U, N> in(shared_input + threadIdx.y * last_dim + j);
				vec<U, N> out = vec<U, N>(alpha) * (gamma * (in - vec<U, N>(avg)) * vec<U, N>(inv_stddev) + beta);
				switch (act)
				{
					case ml::ACTIVATION_SIGMOID:
						out = vectors::sigmoid(out);
						break;
					case ml::ACTIVATION_TANH:
						out = vectors::tanh(out);
						break;
					case ml::ACTIVATION_RELU:
						out = vectors::relu(out);
						break;
					case ml::ACTIVATION_LEAKY_RELU:
						out = select(out > zero<U, N>(), out, out * vec<U, N>(0.1f));
						break;
				}
				if (beta_y != 0.0f)
					out += vec<U, N>(beta_y) * load_vec<U, N>(output + i * last_dim + j);
				store_vec(output + i * last_dim + j, out);
			}
		}
	}

	template<typename T, int N, typename U = T>
	__global__ void kernel_layernorm_backward_v2(const T *input, T *gradient_prev, T *gradient_next, const T *weights, U *weights_update,
			U *bias_update, int first_dim, int last_dim, float alpha, float beta_dx)
	{
		assert(last_dim % N == 0);

		extern __shared__ char shared_array[];

		U *shared_input = reinterpret_cast<U*>(shared_array);
		U *shared_gradient = shared_input + last_dim;
		U *shared_weights = shared_gradient + last_dim;
		U *shared_weights_update = shared_weights + last_dim;
		U *shared_bias_update = shared_weights_update + last_dim;

		__shared__ cg::block_tile_memory<256> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile<256> tile = cg::tiled_partition<256>(thb);

		vec<U, N> thread_weights_update(0.0f);
		vec<U, N> thread_bias_update(0.0f);

		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			const vec<U, N> zero(0.0f);
			zero.store(shared_weights_update + j);
			zero.store(shared_bias_update + j);
			vector_copy<N>(shared_weights + j, weights + j);
		}

		__syncthreads();

		for (int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			U avg = 0.0f;
			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<U, N> in = load_vec<U, N>(input + i * last_dim + j);
				const vec<U, N> grad = load_vec<U, N>(gradient_next + i * last_dim + j);
				avg += horizontal_add(in);
				store_vec(shared_input + j, in);
				store_vec(shared_gradient + j, grad);
			}
			avg = cg::reduce(tile, avg, cg::plus<U>()) / last_dim;

			U var = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
				var += square(shared_input[j] - avg);
			const float inv_stddev = get_inv_stddev(cg::reduce(tile, var, cg::plus<U>()), last_dim, 1.0e-6f);

			U d_sigma = 0.0f;
			U d_mu = 0.0f;
			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				const U in = (shared_input[j] - avg) * inv_stddev;
				const U grad = shared_gradient[j];
				const U gamma = shared_weights[j];

				d_sigma -= grad * in * gamma;
				d_mu -= grad * gamma;
				shared_weights_update[j] += grad * in;
				shared_bias_update[j] += grad;

				shared_input[j] = in;
				shared_gradient[j] = grad * gamma;
			}

			d_sigma = cg::reduce(tile, d_sigma, cg::plus<U>()) * inv_stddev / (last_dim - 1);
			d_mu = cg::reduce(tile, d_mu, cg::plus<U>()) * inv_stddev / last_dim;

			for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
			{
				const vec<U, N> in = load_vec<U, N>(shared_input + j);
				const vec<U, N> grad = load_vec<U, N>(shared_gradient + j);
				vec<U, N> tmp = vec<U, N>(alpha) * (grad * inv_stddev + d_sigma * in + d_mu);
				if (beta_dx != 0.0f)
					tmp += vec<U, N>(beta_dx) * load_vec<U, N>(gradient_prev + i * last_dim + j);
				store_vec(gradient_prev + i * last_dim + j, tmp);
			}
		}
		__syncthreads();
		for (int j = N * threadIdx.x; j < last_dim; j += N * blockDim.x)
		{
			vector_copy<N>(weights_update + blockIdx.x * last_dim + j, shared_weights_update + j);
			vector_copy<N>(bias_update + blockIdx.x * last_dim + j, shared_bias_update + j);
		}
	}

	template<typename T>
	__global__ void kernel_reduce_first_dim(float beta_dw, T *dst, const T *src, int first_dim, int last_dim)
	{
		__shared__ T workspace[32][32 + 1];

		const int last_dim_idx = 32 * blockIdx.x + threadIdx.x;
		if (last_dim_idx < last_dim)
		{
			T local_sum = 0.0f;
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
				local_sum += src[i * last_dim + last_dim_idx];
			workspace[threadIdx.y][threadIdx.x] = local_sum;
		}
		__syncthreads();
		T local_sum = workspace[threadIdx.x][threadIdx.y];

		for (int k = 16; k >= 1; k /= 2)
			local_sum += __shfl_xor_sync(0xffffffff, local_sum, k);
		__syncthreads();
		if (threadIdx.x == 0)
			workspace[0][threadIdx.y] = local_sum;
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < last_dim)
		{
			T tmp = workspace[0][threadIdx.x];
			if (beta_dw != 0.0f)
				tmp += beta_dw * dst[last_dim_idx];
			dst[last_dim_idx] = tmp;
		}
	}

}

namespace ml
{

	void cuda_layernorm_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		dim3 blockDim(32, 8);
		dim3 gridDim(1, std::min(1024u, (first_dim + blockDim.y - 1) / blockDim.y));

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		const int shared_mem = sizeof(float) * (2 + blockDim.y) * last_dim;
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_layernorm_forward_v4<half, 4, float> <<<gridDim, blockDim, shared_mem, stream >>>(data<half>(x), data<half>(y),
							data<half>(w), data<half>(b), first_dim, last_dim, alpha, beta, act);
				else
					kernel_layernorm_forward_v4<half, 1, float> <<<gridDim, blockDim, shared_mem, stream >>>(data<half>(x), data<half>(y),
							data<half>(w), data<half>(b), first_dim, last_dim, alpha, beta, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_layernorm_forward_v4<float, 4> <<<gridDim, blockDim, shared_mem, stream >>>(data<float>(x), data<float>(y), data<float>(w),
							data<float>(b), first_dim, last_dim, alpha, beta, act);
				else
					kernel_layernorm_forward_v4<float, 1> <<<gridDim, blockDim, shared_mem, stream >>>(data<float>(x), data<float>(y), data<float>(w),
							data<float>(b), first_dim, last_dim, alpha, beta, act);
				break;
			}
//			case DTYPE_FLOAT64:
//			{
//				const int shared_mem = sizeof(double) * 3 * last_dim;
//				dim3 blockDim(32, 4);
//				kernel_layernorm_forward_v4<double, 1> <<<gridDim, blockDim, shared_mem, stream >>>(data<double>(x), data<double>(y), data<double>(w),
//						data<double>(b), first_dim, last_dim, alpha, beta, act);
//				break;
//			}
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_layernorm_backward(mlContext_t context, float alpha, const mlTensor_t x, float beta_dx, mlTensor_t dx, mlTensor_t dy,
			const mlTensor_t w, float beta_dw, mlTensor_t dw, mlTensor_t db)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 blockDim(256);

		float *workspace = ml::cuda_backend::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) std::min(first_dim, 512),
				ml::cuda_backend::Context::getWorkspaceSize(context) / (sizeof(float) * 2 * last_dim));
		float *partial_weights_update = workspace;
		float *partial_bias_update = workspace + workspace_first_dim * last_dim;
		const int shared_mem = sizeof(float) * 5 * last_dim;

		dim3 gridDim(workspace_first_dim);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_layernorm_backward_v2<half, 4, float> <<<gridDim, blockDim, shared_mem, stream >>>(data<half>(x), data<half>(dx),
							data<half>(dy), data<half>(w), partial_weights_update, partial_bias_update, first_dim, last_dim, alpha, beta_dx);
				else
					kernel_layernorm_backward_v2<half, 1, float> <<<gridDim, blockDim, shared_mem, stream >>>(data<half>(x), data<half>(dx),
							data<half>(dy), data<half>(w), partial_weights_update, partial_bias_update, first_dim, last_dim, alpha, beta_dx);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_layernorm_backward_v2<float, 4> <<<gridDim, blockDim, shared_mem, stream >>>(data<float>(x), data<float>(dx),
							data<float>(dy), data<float>(w), partial_weights_update, partial_bias_update, first_dim, last_dim, alpha, beta_dx);
				else
					kernel_layernorm_backward_v2<float, 1> <<<gridDim, blockDim, shared_mem, stream >>>(data<float>(x), data<float>(dx),
							data<float>(dy), data<float>(w), partial_weights_update, partial_bias_update, first_dim, last_dim, alpha, beta_dx);
				break;
			}
//			case DTYPE_FLOAT64:
//			{
//				const int shared_mem = sizeof(double) * 3 * last_dim;
//				dim3 blockDim(32, 4);
//				kernel_layernorm_backward_v2<double, 1> <<<gridDim, blockDim, shared_mem, stream >>>(data<double>(x), data<double>(dx),
//						data<double>(dy), data<double>(w), partial_weights_update, partial_bias_update, first_dim, last_dim, alpha, beta_dx);
//				break;
//			}
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2((last_dim + 31) / 32);
		if (dw.data != nullptr)
			kernel_reduce_first_dim<<<gridDim2, blockDim2, 0, stream >>>(beta_dw, data<float>(dw), partial_weights_update, workspace_first_dim,
					last_dim);
		if (db.data != nullptr)
			kernel_reduce_first_dim<<<gridDim2, blockDim2, 0, stream >>>(beta_dw, data<float>(db), partial_bias_update, workspace_first_dim,
					last_dim);

		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */

