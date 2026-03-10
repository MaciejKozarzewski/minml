/*
 * moe.cu
 *
 *  Created on: Feb 6, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "activations.cuh"
#include "../utils.hpp"
#include "../vec/vec_headers.cuh"
#include "../helpers/misc.cuh"
#include "../helpers/indexers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;
	using namespace ml;

	__host__ __device__ int round_up_to_power_of_2(int i) noexcept
	{
		if (i <= 1)
			return 1;
		i--;
		i |= i >> 1;
		i |= i >> 2;
		i |= i >> 4;
		i |= i >> 8;
		i |= i >> 16;
		return i + 1;
	}

	template<typename T>
	__device__ __forceinline__ void swap(T &a, T &b)
	{
		const T tmp = a;
		a = b;
		b = tmp;
	}

	template<typename T>
	__global__ void kernel_select_top_k(const T *input, int *indices, T *values, int top_k, int first_dim, int last_dim)
	{
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
		extern __shared__ char shared_array[];
		int *s_idx = reinterpret_cast<int*>(shared_array);
		T *s_val = reinterpret_cast<T*>(s_idx + blockDim.x);

		const int tid = threadIdx.x;

		s_val[tid] = (tid < last_dim) ? input[blockIdx.x * last_dim + tid] : static_cast<T>(-1000.0f);
		s_idx[tid] = tid;
		__syncthreads();

		// first sort by values to pick top k
		for (int k = 2; k <= blockDim.x; k *= 2)
		{
			for (int j = k / 2; j > 0; j /= 2)
			{
				const int ixj = tid ^ j;
				if (ixj > tid)
				{
					const bool ascending = ((tid & k) == 0);
					if ((s_val[tid] < s_val[ixj]) == ascending)
					{
						swap(s_val[tid], s_val[ixj]);
						swap(s_idx[tid], s_idx[ixj]);
					}
				}
				__syncthreads();
			}
		}

		// mask out entries beyond top k
		if (tid >= top_k)
			s_idx[tid] = last_dim + 1;
		__syncthreads();

		// now sort by indices in ascending order
		const int k_dim = round_up_to_power_of_2(top_k);
		for (int k = 2; k <= k_dim; k *= 2)
		{
			for (int j = k / 2; j > 0; j /= 2)
			{
				const int ixj = tid ^ j;
				if (ixj > tid)
				{
					const bool ascending = ((tid & k) == 0);
					if ((s_idx[tid] > s_idx[ixj]) == ascending)
					{
						swap(s_val[tid], s_val[ixj]);
						swap(s_idx[tid], s_idx[ixj]);
					}
				}
				__syncthreads();
			}
		}

		if (tid < top_k)
		{
			if (indices != nullptr)
				indices[blockIdx.x * top_k + tid] = s_idx[tid];
			if (values != nullptr)
				values[blockIdx.x * top_k + tid] = s_val[tid];
		}
	#endif
	}

	template<typename T>
	__global__ void kernel_hash_routing(T *indices_and_values, int batch_size, int tokens, int experts, int capacity)
	{
		const int b = blockIdx.x;

		const Indexer<4> indexer(batch_size, 2, experts, capacity);

		for (int e = threadIdx.y; e < experts; e += blockDim.y)
			for (int k = threadIdx.x; k < capacity; k += blockDim.x)
			{
				const int token_index = k * experts + e;
				if (token_index < tokens)
				{
					indices_and_values[indexer.at(b, 0, e, k)] = static_cast<T>(token_index);
					indices_and_values[indexer.at(b, 1, e, k)] = static_cast<T>(1.0f);
				}
				else
				{
					indices_and_values[indexer.at(b, 0, e, k)] = static_cast<T>(-1.0f);
					indices_and_values[indexer.at(b, 1, e, k)] = static_cast<T>(0.0f);
				}
			}
	}
	template<typename T, typename U = T>
	__global__ void kernel_softmax_and_top_1(const T *input, const T *bias, int *expert_indices, T *expert_scores, int first_dim, int experts)
	{
		// input	[NHWE]
		// output	[NHW] (int) x2	[NHW] (T)

		extern __shared__ char shared_array[];
		U *expert_bias = reinterpret_cast<U*>(shared_array);
		U *workspace = expert_bias + experts;

		for (int j = threadIdx.y * blockDim.x + threadIdx.x; j < experts; j += blockDim.x * blockDim.y)
			expert_bias[j] = bias[j];
		__syncthreads();

		const Indexer<2> input_indexer(first_dim, experts);
		const Indexer<1> output_indexer(first_dim);

		for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
		{
			U max_logit = static_cast<U>(-1e+32f);
			U max_score = static_cast<U>(-1e+32f);
			for (int j = threadIdx.x; j < experts; j += blockDim.x)
			{
				const U x = input[input_indexer.at(i, j)];
				max_logit = max(max_logit, x);
				max_score = max(max_score, x + expert_bias[j]);
				workspace[threadIdx.y * experts + j] = x;
			}
			for (int k = 16; k >= 1; k /= 2)
			{
				max_logit = max(max_logit, __shfl_xor_sync(0xffffffff, max_logit, k));
				max_score = max(max_score, __shfl_xor_sync(0xffffffff, max_score, k));
			}
			__syncthreads();

			U sum = static_cast<U>(0.0f);
			for (int j = threadIdx.x; j < experts; j += blockDim.x)
				sum += exp(workspace[threadIdx.y * experts + j] - max_logit);
			for (int k = 16; k >= 1; k /= 2)
				sum += __shfl_xor_sync(0xffffffff, sum, k);
			__syncthreads();

			for (int j = threadIdx.x; j < experts; j += blockDim.x)
				if (max_score == (workspace[threadIdx.y * experts + j] + expert_bias[j]))
				{
					const U x = exp(workspace[threadIdx.y * experts + j] - max_logit) / sum;
					expert_indices[i] = j;
					expert_scores[i] = static_cast<T>(x);
				}
			__syncthreads();
		}
	}
	template<typename T>
	__global__ void kernel_token_choice_routing(const int *indices, const T *scores, T *indices_and_values, int batch_size, int tokens, int experts,
			int capacity)
	{
		// input	[NHWE]
		// indices	[N2EK]
		extern __shared__ char shared_array[];
		const int workspace_size = round_up_to_power_of_2(tokens);
		int *expert_count = reinterpret_cast<int*>(shared_array);
		int *expert_index = reinterpret_cast<int*>(expert_count + experts + 1);
		int *token_index = reinterpret_cast<int*>(expert_index + workspace_size);
		T *expert_score = reinterpret_cast<T*>(token_index + workspace_size);

		const int b = blockIdx.x;

		const Indexer<2> input_indexer(batch_size, tokens);
		for (int j = threadIdx.x; j < workspace_size; j += blockDim.x)
			if (j < tokens)
			{
				expert_index[j] = indices[input_indexer.at(b, j)];
				token_index[j] = j;
				expert_score[j] = scores[input_indexer.at(b, j)];
			}
			else
			{
				expert_index[j] = experts + 1;
				token_index[j] = -1;
				expert_score[j] = static_cast<T>(0.0f);
			}
		for (int j = threadIdx.x; j <= experts; j += blockDim.x)
			expert_count[j] = 0;
		__syncthreads();

		// first sort by values to pick top k
		const int tid = threadIdx.x;
		for (int k = 2; k <= blockDim.x; k *= 2)
		{
			for (int j = k / 2; j > 0; j /= 2)
			{
				const int ixj = tid ^ j;
				if (ixj > tid)
				{
					const bool ascending = ((tid & k) == 0);
					// sort by expert id (ascending), and by score within same expert id (descending)
					const bool same_expert = (expert_index[tid] == expert_index[ixj]);
					const bool compare = same_expert ? (expert_score[tid] <= expert_score[ixj]) : (expert_index[tid] > expert_index[ixj]);
					if (compare == ascending)
					{
						swap(expert_index[tid], expert_index[ixj]);
						swap(token_index[tid], token_index[ixj]);
						swap(expert_score[tid], expert_score[ixj]);
					}
				}
				__syncthreads();
			}
		}

		// now count how many times each expert is selected
		for (int j = threadIdx.x; j < tokens; j += blockDim.x)
			for (int e = expert_index[j] + 1; e <= experts; e++)
				atomicAdd(expert_count + e, 1);
		__syncthreads();

		const int eidx = tid / 32;
		const int cidx = tid % 32;
		const int e_str = blockDim.x / 32;

		const Indexer<4> output_indexer(batch_size, 2, experts, capacity);
		for (int e = eidx; e < experts; e += e_str)
		{
			const int c0 = expert_count[e];
			const int c1 = expert_count[e + 1];
			for (int c = cidx; c < capacity; c += 32)
				if (c < (c1 - c0))
				{
					indices_and_values[output_indexer.at(b, 0, e, c)] = static_cast<T>(token_index[c0 + c]);
					indices_and_values[output_indexer.at(b, 1, e, c)] = static_cast<T>(expert_score[c0 + c]);
				}
				else
				{
					indices_and_values[output_indexer.at(b, 0, e, c)] = static_cast<T>(-1.0f);
					indices_and_values[output_indexer.at(b, 1, e, c)] = static_cast<T>(0.0f);
				}
		}

	}
	template<typename T, typename U = T>
	__global__ void kernel_token_choice_scatter_gradient(T *workspace, const T *indices_and_values, const T *gradient_next, int batch_size,
			int tokens, int experts, int capacity)
	{
		const Indexer<3> input_indexer(batch_size, tokens, experts);
		const Indexer<4> output_indexer(batch_size, 2, experts, capacity);

		const int b = blockIdx.x;

		for (int e = threadIdx.y; e < experts; e += blockDim.y)
			for (int k = threadIdx.x; k < capacity; k += blockDim.x)
			{
				const int index = (int) indices_and_values[output_indexer.at(b, 0, e, k)];
				if (index >= 0)
					workspace[input_indexer.at(b, index, e)] = gradient_next[output_indexer.at(b, 1, e, k)];
			}
	}
	template<typename T, typename U = T>
	__global__ void kernel_token_choice_backward(float beta, T *gradient_prev, const T *input, const T *gradient_next, int batch_size, int tokens,
			int experts, int capacity)
	{
		// input	[NHWE]
		// output	[N2EK]

		extern __shared__ char shared_array[];
		U *workspace = reinterpret_cast<U*>(shared_array);

		const Indexer<3> input_indexer(batch_size, tokens, experts);
		const Indexer<4> output_indexer(batch_size, 2, experts, capacity);

		const int b = blockIdx.x;

		for (int t = threadIdx.y; t < tokens; t += blockDim.y)
		{
			U max_value = static_cast<U>(-1e+32f);
			for (int e = threadIdx.x; e < experts; e += blockDim.x)
			{
				const U x = input[input_indexer.at(b, t, e)];
				max_value = max(max_value, x);
				workspace[threadIdx.y * experts + e] = x;
			}
			for (int k = 16; k >= 1; k /= 2)
				max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, k));
			__syncthreads();

			U sum = static_cast<U>(0.0f);
			for (int e = threadIdx.x; e < experts; e += blockDim.x)
			{
				const U x = exp(workspace[threadIdx.y * experts + e] - max_value);
				workspace[threadIdx.y * experts + e] = x;
				sum += x;
			}
			for (int k = 16; k >= 1; k /= 2)
				sum += __shfl_xor_sync(0xffffffff, sum, k);
			__syncthreads();

			const U inv_scale = static_cast<U>(1.0f) / sum;
			for (int e = threadIdx.x; e < experts; e += blockDim.x)
				workspace[threadIdx.y * experts + e] *= inv_scale;
			__syncthreads();

			U tmp = static_cast<U>(0.0f);
			for (int e = threadIdx.x; e < experts; e += blockDim.x)
			{
				const U y = workspace[threadIdx.y * experts + e];
				const U dy = static_cast<U>(gradient_next[input_indexer.at(b, t, e)]);
				tmp += dy * y;
			}
			for (int k = 16; k >= 1; k /= 2)
				tmp += __shfl_xor_sync(0xffffffff, tmp, k);
			__syncthreads();

			for (int e = threadIdx.x; e < experts; e += blockDim.x)
			{
				const int idx = input_indexer.at(b, t, e);
				const U y = workspace[threadIdx.y * experts + e];
				const U dy = static_cast<U>(gradient_next[idx]);
				U dx = y * (dy - tmp);
				if (beta != 0.0f)
					dx += static_cast<U>(beta) * static_cast<U>(gradient_prev[idx]);
				gradient_prev[idx] = static_cast<T>(dx);
			}
			__syncthreads();
		}
	}

	template<typename T, int N>
	__global__ void kernel_gather_forward(const T *input, float beta, T *output, const T *indices_and_values, int batch_size, int tokens,
			int channels, int experts, int top_k)
	{
		assert(channels % N == 0);
		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<3> input_indexer(batch_size, tokens, channels);
		const Indexer<4> output_indexer(batch_size, top_k, experts, channels);
		const Indexer<4> indices_indexer(batch_size, 2, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices_and_values[indices_indexer.at(b, 0, e, k)];
			if (token_index >= 0)
			{
				const T *src = input + input_indexer.at(b, token_index, 0);
				T *dst = output + output_indexer.at(b, k, e, 0);
				for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
				{
					vec<T, N> tmp(src + c);
					if (beta != 0.0f)
						tmp += vec<T, N>(beta) * vec<T, N>(dst + c);
					tmp.store(dst + c);
				}
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_gather_backward(const T *gradient_next, T *gradient_prev, const T *indices_and_values, int batch_size, int tokens,
			int channels, int experts, int top_k)
	{
		assert(channels % N == 0);
		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<3> prev_indexer(batch_size, tokens, channels);
		const Indexer<4> next_indexer(batch_size, top_k, experts, channels);
		const Indexer<4> indices_indexer(batch_size, 2, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices_and_values[indices_indexer.at(b, 0, e, k)];
			if (token_index >= 0)
			{
				const T *src = gradient_next + next_indexer.at(b, k, e, 0);
				T *dst = gradient_prev + prev_indexer.at(b, token_index, 0);
				for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
					atomic_add(dst + c, vec<T, N>(src + c));
			}
		}
	}

	template<typename T, int N>
	__global__ void kernel_scatter_forward(const T *input, T *output, const T *indices_and_values, int batch_size, int tokens, int channels,
			int experts, int top_k)
	{
		assert(channels % N == 0);
		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<4> input_indexer(batch_size, top_k, experts, channels);
		const Indexer<3> output_indexer(batch_size, tokens, channels);
		const Indexer<4> indices_indexer(batch_size, 2, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices_and_values[indices_indexer.at(b, 0, e, k)];
			if (token_index >= 0)
			{
				const vec<T, N> scale(indices_and_values[indices_indexer.at(b, 1, e, k)]);
				const T *src = input + input_indexer.at(b, k, e, 0);
				T *dst = output + output_indexer.at(b, token_index, 0);
				for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
					atomic_add(dst + c, vec<T, N>(src + c) * scale);
			}
		}
	}
	template<typename T, int N, typename U = T>
	__global__ void kernel_scatter_backward(const T *gradient_next, float beta_prev, T *gradient_prev, const T *input, const T *indices_and_values,
			float beta_scales, T *gradient_scales, int batch_size, int tokens, int channels, int experts, int top_k)
	{
		assert(channels % N == 0);
		extern __shared__ char shared_array[];
		U *workspace_grad = reinterpret_cast<U*>(shared_array);
		U *workspace_scales = workspace_grad + top_k;

		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<4> indices_indexer(batch_size, 2, experts, top_k);

		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int k = tid; k < top_k; k += blockDim.x * blockDim.y)
		{
			workspace_grad[k] = static_cast<T>(0.0f);
			workspace_scales[k] = static_cast<T>(indices_and_values[indices_indexer.at(b, 1, e, k)]);
		}
		__syncthreads();

		const Indexer<4> prev_indexer(batch_size, top_k, experts, channels);
		const Indexer<3> next_indexer(batch_size, tokens, channels);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			vec<T, N> scale_gradient(static_cast<T>(0.0f));

			const int token_index = indices_and_values[indices_indexer.at(b, 0, e, k)];
			if (token_index >= 0)
			{
				const vec<T, N> scale(workspace_scales[k]);

				for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
				{
					const vec<T, N> dy(gradient_next + next_indexer.at(b, token_index, c));
					const vec<T, N> x(input + prev_indexer.at(b, k, e, c));
					vec<T, N> dx = dy * scale;
					scale_gradient += dy * x;
					if (beta_prev != 0.0f)
						dx += vec<T, N>(beta_prev) * vec<T, N>(gradient_prev + prev_indexer.at(b, k, e, c));
					dx.store(gradient_prev + prev_indexer.at(b, k, e, c));
				}
			}

			U reduced_scale_gradient = horizontal_add(scale_gradient);
			for (int i = 16; i >= 1; i /= 2)
				reduced_scale_gradient += __shfl_xor_sync(0xffffffff, reduced_scale_gradient, i);
			if (threadIdx.x == 0)
				workspace_grad[k] = reduced_scale_gradient;
			__syncthreads();
		}

		for (int k = tid; k < top_k; k += blockDim.x * blockDim.y)
		{
			const int idx = indices_indexer.at(b, 1, e, k);
			T tmp = workspace_grad[k];
			if (beta_scales != 0.0f)
				tmp += gradient_scales[idx] * static_cast<T>(beta_scales);
			gradient_scales[idx] = tmp;
			gradient_scales[indices_indexer.at(b, 0, e, k)] = static_cast<T>(0.0f); // set index gradient to zero
		}
	}

	int get_batch_size(const mlTensor_t &t)
	{
		assert(t.rank == 3 || t.rank == 4);
		return (t.rank == 3) ? t.dim[0] : t.dim[2]; // ECC : NKEC
	}
	int get_batch_stride(const mlTensor_t &t)
	{
		assert(t.rank == 3 || t.rank == 4);
		return (t.rank == 3) ? t.dim[1] * t.dim[2] : t.dim[3]; // ECC : NKEC
	}
	int get_ld(const mlTensor_t &t)
	{
		assert(t.rank == 3 || t.rank == 4);
		return (t.rank == 3) ? t.dim[2] : t.dim[2] * t.dim[3]; // ECC : NKEC
	}
	int num_rows(const mlTensor_t &t)
	{
		assert(t.rank == 3 || t.rank == 4);
		return (t.rank == 3) ? t.dim[1] : t.dim[0] * t.dim[1]; // ECC : NKEC
	}
	int num_cols(const mlTensor_t &t)
	{
		assert(t.rank == 3 || t.rank == 4);
		return (t.rank == 3) ? t.dim[2] : t.dim[3]; // ECC : NKEC
	}
	void batch_gemm(mlContext_t context, char opA, char opB, float alpha, const mlTensor_t A, const mlTensor_t B, float beta, mlTensor_t C)
	{
		const int batch = get_batch_size(A);
		cublasOperation_t op_A = is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_B = is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		const int M = is_transpose(opB) ? num_rows(B) : num_cols(B);
		const int N = is_transpose(opA) ? num_cols(A) : num_rows(A);
		const int K = is_transpose(opB) ? num_cols(B) : num_rows(B);

		const int LDA = get_ld(A);
		const int LDB = get_ld(B);
		const int LDC = get_ld(C);
		const int strideA = get_batch_stride(A);
		const int strideB = get_batch_stride(B);
		const int strideC = get_batch_stride(C);

		cublasHandle_t handle = ml::cuda_backend::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (C.dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(is_fp16(A));
				assert(is_fp16(B));
				const half _alpha = alpha;
				const half _beta = beta;
				cublasStatus_t status = cublasHgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, data<half>(B), LDB, strideB, data<half>(A),
						LDA, strideA, &_beta, data<half>(C), LDC, strideC, batch);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
			case DTYPE_FLOAT32:
			{
				assert(A.dtype == B.dtype);
				assert(is_fp32(A) || is_fp16(A));
				const float _alpha = alpha;
				const float _beta = beta;
				if (is_fp32(A))
				{
					if (ml::cuda_backend::Context::allowsTF32(context))
					{
						cublasStatus_t status = cublasGemmStridedBatchedEx(handle, op_B, op_A, M, N, K, &_alpha, B.data, CUDA_R_32F, LDB, strideB,
								A.data, CUDA_R_32F, LDA, strideA, &_beta, C.data, CUDA_R_32F, LDC, strideC, batch, CUBLAS_COMPUTE_32F_FAST_TF32,
								CUBLAS_GEMM_DEFAULT);
						assert(status == CUBLAS_STATUS_SUCCESS);
					}
					else
					{
						cublasStatus_t status = cublasSgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, data<float>(B), LDB, strideB,
								data<float>(A), LDA, strideA, &_beta, data<float>(C), LDC, strideC, batch);
						assert(status == CUBLAS_STATUS_SUCCESS);
					}
				}
				else
				{
					cublasStatus_t status = cublasGemmStridedBatchedEx(handle, op_B, op_A, M, N, K, &_alpha, B.data, CUDA_R_16F, LDB, strideB, A.data,
							CUDA_R_16F, LDA, strideA, &_beta, C.data, CUDA_R_32F, LDC, strideC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				break;
			}
			case DTYPE_FLOAT64:
			{
				const double _alpha = alpha;
				const double _beta = beta;
				cublasStatus_t status = cublasDgemmStridedBatched(handle, op_B, op_A, M, N, K, &_alpha, data<double>(B), LDB, strideB,
						data<double>(A), LDA, strideA, &_beta, data<double>(C), LDC, strideC, batch);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}

	template<typename T, int N>
	__global__ void kernel_batch_add_bias_act(T *output, const T *bias, int first_dim, int last_dim, mlActivationType_t act)
	{
		assert(last_dim % N == 0);
		const Indexer<2> bias_indexer(gridDim.z, last_dim);
		const Indexer<3> in_out_indexer(first_dim, gridDim.z, last_dim);

		const int expert_idx = blockIdx.z;

		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * N; j < last_dim; j += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> _bias(bias + bias_indexer.at(expert_idx, j));
			for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += gridDim.y * blockDim.y)
			{
				const int offset = in_out_indexer.at(i, expert_idx, j);
				vec<T, N> tmp(output + offset);
				tmp = activation_forward(act, tmp + _bias);
				tmp.store(output + offset);
			}
		}
	}
	template<typename T, int N, typename U = T>
	__global__ void kernel_act_bias_backward(T *gradient_next, const T *output, T *bias_gradient, int first_dim, int last_dim,
			ml::mlActivationType_t act)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 8);
		assert(last_dim % N == 0);
		__shared__ vec<U, N> workspace[8 * 32];

		const int expert_idx = blockIdx.z;

		const Indexer<3> in_out_indexer(first_dim, gridDim.z, last_dim);

		const int last_dim_idx = N * (blockDim.x * blockIdx.x + threadIdx.x);
		vec<U, N> local_sum(0.0f);
		if (last_dim_idx < last_dim)
		{
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				const int tmp_idx = in_out_indexer.at(i, expert_idx, last_dim_idx);
				vec<U, N> dy = load_vec<U, N>(gradient_next + tmp_idx);
				if (act != ml::ACTIVATION_LINEAR)
				{
					const vec<U, N> y = load_vec<U, N>(output + tmp_idx);
					switch (act)
					{
						case ml::ACTIVATION_SIGMOID:
							dy *= y * (one<U, N>() - y);
							break;
						case ml::ACTIVATION_TANH:
							dy *= one<U, N>() - square(y);
							break;
						case ml::ACTIVATION_RELU:
							dy = select(y > zero<U, N>(), dy, zero<U, N>());
							break;
						case ml::ACTIVATION_LEAKY_RELU:
							dy = select(y > zero<U, N>(), dy, dy * vec<U, N>(0.1f));
							break;
					}
					store_vec(gradient_next + tmp_idx, dy);
				}
				local_sum += dy;
			}
		}

		if (bias_gradient != nullptr)
		{
			const Indexer<2> workspace_indexer(8, 32);
			workspace[workspace_indexer.at(threadIdx.y, threadIdx.x)] = local_sum;
			__syncthreads();

			for (int i = blockDim.y / 2; i >= 1; i /= 2)
			{
				if (threadIdx.y < i)
					workspace[workspace_indexer.at(threadIdx.y, threadIdx.x)] += workspace[workspace_indexer.at(i + threadIdx.y, threadIdx.x)];
				__syncthreads();
			}

			const Indexer<2> db_indexer(gridDim.z, last_dim);
			if (threadIdx.y == 0 && last_dim_idx < last_dim)
			{
				const vec<T, N> tmp = convert<T>(workspace[workspace_indexer.at(0, threadIdx.x)]);
				atomic_add(bias_gradient + db_indexer.at(expert_idx, last_dim_idx), tmp);
			}
		}
	}

	template<typename T>
	__global__ void kernel_expert_bias_update(const T *indices_and_values, int *workspace, int batch_size, int experts, int capacity)
	{
		const int b = blockIdx.x;
		const Indexer<4> indexer(batch_size, 2, experts, capacity);

		for (int e = threadIdx.y; e < experts; e += blockDim.y)
		{
			int token_count = 0;
			for (int k = threadIdx.x; k < capacity; k += blockDim.x)
				token_count += static_cast<int>(indices_and_values[indexer.at(b, 0, e, k)]) != -1;
			for (int k = 16; k >= 1; k /= 2)
				token_count += __shfl_xor_sync(0xffffffff, token_count, k);
			__syncthreads();
			if (threadIdx.x == 0)
				atomicAdd(workspace + e, token_count);
			__syncthreads();
		}
	}
	template<typename T>
	__global__ void kernel_expert_bias_update(const int *workspace, float alpha, T *bias, int batch_size, int tokens, int experts)
	{
		const float avg_capacity = static_cast<float>(tokens) / static_cast<float>(experts);
		for (int e = threadIdx.x; e < experts; e += blockDim.x)
		{
			const float token_count = static_cast<float>(workspace[e]) / static_cast<float>(batch_size);
			if (token_count > avg_capacity)
				bias[e] -= alpha;
			else
				bias[e] += alpha;
		}
	}
}

namespace ml
{
	void cuda_hash_routing(mlContext_t context, const mlTensor_t x, mlTensor_t indices_and_values)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = indices_and_values.dim[0];
		assert(indices_and_values.dim[1] == 2);
		const int experts = indices_and_values.dim[2];
		const int capacity = indices_and_values.dim[3];
		assert(x.dim[0] == batch_size);
		const int tokens = x.dim[1] * x.dim[2];

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_hash_routing<<<grid_dim, block_dim, 0, stream>>>(data<half>(indices_and_values), batch_size, tokens, experts, capacity);
				break;
			case DTYPE_FLOAT32:
				kernel_hash_routing<<<grid_dim, block_dim, 0, stream>>>(data<float>(indices_and_values), batch_size, tokens, experts, capacity);
				break;
			case DTYPE_FLOAT64:
				kernel_hash_routing<<<grid_dim, block_dim, 0, stream>>>(data<double>(indices_and_values), batch_size, tokens, experts, capacity);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_token_choice_routing_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t bias, mlTensor_t indices_and_values)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = indices_and_values.dim[0];
		assert(indices_and_values.dim[1] == 2);
		const int experts = indices_and_values.dim[2];
		const int capacity = indices_and_values.dim[3];
		assert(x.dim[0] == batch_size);
		const int tokens = x.dim[1] * x.dim[2];

		int *expert_indices = getPointer<int>(ml::cuda_backend::Context::getWorkspace(context));
		void *expert_scores = reinterpret_cast<void*>(expert_indices + batch_size * tokens + (batch_size * tokens) % 2);

		dim3 block_dim(32, 8);
		dim3 grid_dim(min(1024, batch_size * tokens));

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				const int shared_mem_size = (block_dim.y + 1) * experts * sizeof(float); // +1 for storing expert biases
				kernel_softmax_and_top_1<half, float> <<<grid_dim, block_dim, shared_mem_size, stream>>>(data<half>(x), data<half>(bias),
						expert_indices, reinterpret_cast<half*>(expert_scores), batch_size * tokens, experts);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const int shared_mem_size = (block_dim.y + 1) * experts * sizeof(float);
				kernel_softmax_and_top_1<<<grid_dim, block_dim, shared_mem_size, stream>>>(data<float>(x), data<float>(bias), expert_indices,
						reinterpret_cast<float*>(expert_scores), batch_size * tokens, experts);
				break;
			}
			case DTYPE_FLOAT64:
			{
				const int shared_mem_size = (block_dim.y + 1) * experts * sizeof(double);
				kernel_softmax_and_top_1<<<grid_dim, block_dim, shared_mem_size, stream>>>(data<double>(x), data<double>(bias), expert_indices,
						reinterpret_cast<double*>(expert_scores), batch_size * tokens, experts);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 block_dim2(round_up_to_power_of_2(tokens));
		dim3 grid_dim2(min(1024, batch_size));

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				const int shared_mem_size = block_dim2.x * (2 * sizeof(int) + sizeof(half)) + (experts + 1) * sizeof(int);
				kernel_token_choice_routing <<<grid_dim2, block_dim2, shared_mem_size, stream>>>(expert_indices,
						reinterpret_cast<half*>(expert_scores), data<half>(indices_and_values), batch_size, tokens, experts, capacity);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const int shared_mem_size = block_dim2.x * (2 * sizeof(int) + sizeof(float)) + (experts + 1) * sizeof(int);
				kernel_token_choice_routing<<<grid_dim2, block_dim2, shared_mem_size, stream>>>(expert_indices,
						reinterpret_cast<float*>(expert_scores), data<float>(indices_and_values), batch_size, tokens, experts, capacity);
				break;
			}
			case DTYPE_FLOAT64:
			{
				const int shared_mem_size = block_dim2.x * (2 * sizeof(int) + sizeof(double)) + (experts + 1) * sizeof(int);
				kernel_token_choice_routing<<<grid_dim2, block_dim2, shared_mem_size, stream>>>(expert_indices,
						reinterpret_cast<double*>(expert_scores), data<double>(indices_and_values), batch_size, tokens, experts, capacity);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_token_choice_routing_backward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices_and_values, const mlTensor_t dy,
			float beta, mlTensor_t dx, float alpha, mlTensor_t bias, mlTensor_t workspace)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = indices_and_values.dim[0];
		assert(indices_and_values.dim[1] == 2);
		const int experts = indices_and_values.dim[2];
		const int capacity = indices_and_values.dim[3];
		assert(x.dim[0] == batch_size);
		const int tokens = x.dim[1] * x.dim[2];

		dim3 block_dim(32, 8);
		dim3 grid_dim(min(1024, batch_size));

		cuda_memset(context, workspace.data, 0, volume(workspace) * size_of(workspace.dtype), nullptr, 0);

		int *token_counts = getPointer<int>(ml::cuda_backend::Context::getWorkspace(context));

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				kernel_token_choice_scatter_gradient<<<grid_dim, block_dim, 0, stream>>>(data<half>(workspace), data<half>(indices_and_values),
						data<half>(dy), batch_size, tokens, experts, capacity);
				const int shared_mem_size = block_dim.y * tokens * sizeof(float);
				kernel_token_choice_backward<half, float> <<<grid_dim, block_dim, shared_mem_size, stream>>>(beta, data<half>(dx), data<half>(x),
						data<half>(workspace), batch_size, tokens, experts, capacity);

				kernel_expert_bias_update<<<grid_dim, block_dim, 0, stream>>>(data<half>(indices_and_values), token_counts, batch_size, experts,
						capacity);
				kernel_expert_bias_update<<<1, 256, 0, stream>>>(token_counts, alpha, data<half>(bias), batch_size, tokens, experts);
				break;
			}
			case DTYPE_FLOAT32:
			{
				kernel_token_choice_scatter_gradient<<<grid_dim, block_dim, 0, stream>>>(data<float>(workspace), data<float>(indices_and_values),
						data<float>(dy), batch_size, tokens, experts, capacity);
				const int shared_mem_size = block_dim.y * tokens * sizeof(float);
				kernel_token_choice_backward <<<grid_dim, block_dim, shared_mem_size, stream>>>(beta, data<float>(dx), data<float>(x),
						data<float>(workspace), batch_size, tokens, experts, capacity);

				kernel_expert_bias_update<<<grid_dim, block_dim, 0, stream>>>(data<float>(indices_and_values), token_counts, batch_size, experts,
						capacity);
				kernel_expert_bias_update<<<1, 256, 0, stream>>>(token_counts, alpha, data<float>(bias), batch_size, tokens, experts);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_token_choice_scatter_gradient<<<grid_dim, block_dim, 0, stream>>>(data<double>(workspace), data<double>(indices_and_values),
						data<double>(dy), batch_size, tokens, experts, capacity);
				const int shared_mem_size = block_dim.y * tokens * sizeof(double);
				kernel_token_choice_backward<<<grid_dim, block_dim, shared_mem_size, stream>>>(beta, data<double>(dx), data<double>(x),
						data<double>(workspace), batch_size, tokens, experts, capacity);

				kernel_expert_bias_update<<<grid_dim, block_dim, 0, stream>>>(data<double>(indices_and_values), token_counts, batch_size, experts,
						capacity);
				kernel_expert_bias_update<<<1, 256, 0, stream>>>(token_counts, alpha, data<double>(bias), batch_size, tokens, experts);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_expert_choice_routing(mlContext_t context, const mlTensor_t x, mlTensor_t indices_and_values)
	{
	}

	void cuda_select_top_k(mlContext_t context, const mlTensor_t x, mlTensor_t indices, mlTensor_t values)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		const int top_k = get_last_dim(indices);
		assert(top_k <= last_dim);
		assert(indices.rank == x.rank);

		const int power_of_2 = round_up_to_power_of_2(last_dim);
		const int shared_memory_size = power_of_2 * (size_of(x.dtype) + sizeof(int));

		dim3 block_dim(power_of_2);
		dim3 grid_dim(first_dim);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
				kernel_select_top_k<<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(x), data<int>(indices), data<half>(values), top_k,
						first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_select_top_k<<<grid_dim, block_dim, shared_memory_size, stream>>>(data<float>(x), data<int>(indices), data<float>(values),
						top_k, first_dim, last_dim);
				break;
			case DTYPE_FLOAT64:
				kernel_select_top_k<<<grid_dim, block_dim, shared_memory_size, stream>>>(data<double>(x), data<int>(indices), data<double>(values),
						top_k, first_dim, last_dim);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_gather_tokens_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices_and_values, float beta, mlTensor_t y)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = x.dim[0];
		const int tokens = x.dim[1] * x.dim[2];
		const int channels = x.dim[3];

		assert(batch_size == y.dim[0]);
		const int top_k = y.dim[1];
		const int experts = y.dim[2];
		assert(channels == y.dim[3]);

		assert(indices_and_values.dim[0] == batch_size);
		assert(indices_and_values.dim[1] == 2);
		assert(indices_and_values.dim[2] == experts);
		assert(indices_and_values.dim[3] == top_k);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_gather_forward<half, 4> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), beta, data<half>(y),
							data<half>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_forward<half, 1> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), beta, data<half>(y),
							data<half>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_gather_forward<float, 4> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), beta, data<float>(y),
							data<float>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_forward<float, 1> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), beta, data<float>(y),
							data<float>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_gather_forward<double, 1> <<<grid_dim, block_dim, 0, stream>>>(data<double>(x), beta, data<double>(y),
						data<double>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_gather_tokens_backward(mlContext_t context, const mlTensor_t dy, const mlTensor_t indices_and_values, float beta, mlTensor_t dx)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = dx.dim[0];
		const int tokens = dx.dim[1] * dx.dim[2];
		const int channels = dx.dim[3];

		assert(batch_size == dy.dim[0]);
		const int top_k = dy.dim[1];
		const int experts = dy.dim[2];
		assert(channels == dy.dim[3]);

		assert(indices_and_values.dim[0] == batch_size);
		assert(indices_and_values.dim[1] == 2);
		assert(indices_and_values.dim[2] == experts);
		assert(indices_and_values.dim[3] == top_k);

		cuda_scale_tensor(context, beta, dx);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_gather_backward<half, 4> <<<grid_dim, block_dim, 0, stream>>>(data<half>(dy), data<half>(dx),
							data<half>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_backward<half, 1> <<<grid_dim, block_dim, 0, stream>>>(data<half>(dy), data<half>(dx),
							data<half>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_gather_backward<float, 4> <<<grid_dim, block_dim, 0, stream>>>(data<float>(dy), data<float>(dx),
							data<float>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_backward<float, 1> <<<grid_dim, block_dim, 0, stream>>>(data<float>(dy), data<float>(dx),
							data<float>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_gather_backward<double, 1> <<<grid_dim, block_dim, 0, stream>>>(data<double>(dy), data<double>(dx),
						data<double>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_scatter_tokens_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices_and_values, float beta, mlTensor_t y)
	{
		// x		[NKEC]
		// indices	[N2EK]
		// y		[NHWC]	HW = tokens
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = y.dim[0];
		const int tokens = y.dim[1] * y.dim[2];
		const int channels = y.dim[3];

		assert(batch_size == x.dim[0]);
		const int top_k = x.dim[1];
		const int experts = x.dim[2];
		assert(channels == x.dim[3]);

		assert(batch_size == indices_and_values.dim[0]);
		assert(2 == indices_and_values.dim[1]);
		assert(experts == indices_and_values.dim[2]);
		assert(top_k == indices_and_values.dim[3]);

		cuda_scale_tensor(context, beta, y);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_scatter_forward<half, 4> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), data<half>(y), data<half>(indices_and_values),
							batch_size, tokens, channels, experts, top_k);
				else
					kernel_scatter_forward<half, 1> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), data<half>(y), data<half>(indices_and_values),
							batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_scatter_forward<float, 4> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), data<float>(y),
							data<float>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				else
					kernel_scatter_forward<float, 1> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), data<float>(y),
							data<float>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_scatter_forward<double, 1> <<<grid_dim, block_dim, 0, stream>>>(data<double>(x), data<double>(y),
						data<double>(indices_and_values), batch_size, tokens, channels, experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_scatter_tokens_backward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices_and_values, const mlTensor_t dy, float beta1,
			mlTensor_t dx, float beta2, mlTensor_t dscales)
	{
		// x		[NKEC]
		// indices	[N2EK]
		// dy		[NHWC]	HW = tokens
		// dx		[NKEC]
		// dscales	[NEHW]
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = dy.dim[0];
		const int tokens = dy.dim[1] * dy.dim[2];
		const int channels = dy.dim[3];

		assert(batch_size == x.dim[0]);
		const int top_k = x.dim[1];
		const int experts = x.dim[2];
		assert(channels == x.dim[3]);

		assert(batch_size == indices_and_values.dim[0]);
		assert(2 == indices_and_values.dim[1]);
		assert(experts == indices_and_values.dim[2]);
		assert(top_k == indices_and_values.dim[3]);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				const size_t shared_memory_size = 2 * sizeof(float) * top_k;
				if (channels % 4 == 0)
					kernel_scatter_backward<half, 4, float> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(dy), beta1,
							data<half>(dx), data<half>(x), data<half>(indices_and_values), beta2, data<half>(dscales), batch_size, tokens, channels,
							experts, top_k);
				else
					kernel_scatter_backward<half, 1, float> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(dy), beta1,
							data<half>(dx), data<half>(x), data<half>(indices_and_values), beta2, data<half>(dscales), batch_size, tokens, channels,
							experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const size_t shared_memory_size = 2 * sizeof(float) * top_k;
				if (channels % 4 == 0)
					kernel_scatter_backward<float, 4> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<float>(dy), beta1, data<float>(dx),
							data<float>(x), data<float>(indices_and_values), beta2, data<float>(dscales), batch_size, tokens, channels, experts,
							top_k);
				else
					kernel_scatter_backward<float, 1> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<float>(dy), beta1, data<float>(dx),
							data<float>(x), data<float>(indices_and_values), beta2, data<float>(dscales), batch_size, tokens, channels, experts,
							top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				const size_t shared_memory_size = 2 * sizeof(double) * top_k;
				kernel_scatter_backward<double, 1> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<double>(dy), beta1, data<double>(dx),
						data<double>(x), data<double>(indices_and_values), beta2, data<double>(dscales), batch_size, tokens, channels, experts,
						top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_moe_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta, mlTensor_t y,
			mlActivationType_t act)
	{
		assert(x.rank == 4);
		assert(y.rank == 4);
		assert(b.rank == 2);
		batch_gemm(context, 'n', 't', 1.0f, x, w, beta, y);

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int first_dim = y.dim[0] * x.dim[1];
		const int experts = y.dim[2];
		const int last_dim = y.dim[3];
		dim3 blockDim(32, 8);
		dim3 gridDim_x1((last_dim + 31) / 32, std::min(1024, (first_dim + 7) / 8), experts);
		dim3 gridDim_x4((last_dim + 127) / 128, std::min(1024, (first_dim + 7) / 8), experts);

		switch (x.dtype)
		{
			case ml::DTYPE_FLOAT16:
				if (last_dim % 4 == 0)
					kernel_batch_add_bias_act<half, 4> <<<gridDim_x4, blockDim, 0, stream>>>(data<half>(y), data<half>(b), first_dim, last_dim, act);
				else
					kernel_batch_add_bias_act<half, 1> <<<gridDim_x1, blockDim, 0, stream>>>(data<half>(y), data<half>(b), first_dim, last_dim, act);
				break;
			case ml::DTYPE_FLOAT32:
				if (last_dim % 4 == 0)
					kernel_batch_add_bias_act<float, 4> <<<gridDim_x4, blockDim, 0, stream>>>(data<float>(y), data<float>(b), first_dim, last_dim,
							act);
				else
					kernel_batch_add_bias_act<float, 1> <<<gridDim_x1, blockDim, 0, stream>>>(data<float>(y), data<float>(b), first_dim, last_dim,
							act);
				break;
			case ml::DTYPE_FLOAT64:
				kernel_batch_add_bias_act<double, 1> <<<gridDim_x1, blockDim, 0, stream>>>(data<double>(y), data<double>(b), first_dim, last_dim,
						act);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_moe_backward(mlContext_t context, const mlTensor_t x, const mlTensor_t y, const mlTensor_t w, mlTensor_t dy, float beta_dx,
			mlTensor_t dx, float beta_dw, mlTensor_t dw, float beta_db, mlTensor_t db, mlActivationType_t act)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int first_dim = dy.dim[0] * dy.dim[1];
		const int experts = dy.dim[2];
		const int last_dim = dy.dim[3];

		const int nk_dim = std::min(1024, (first_dim + 7) / 8);

		dim3 blockDim1(32, 8);
		dim3 gridDim1_x1((last_dim + 31) / 32, nk_dim, experts);
		dim3 gridDim1_x4((last_dim + 127) / 127, nk_dim, experts);

		cuda_scale_tensor(context, beta_db, db);

		switch (y.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_act_bias_backward<half, 4, float> <<<gridDim1_x4, blockDim1, 0, stream>>>(data<half>(dy), data<half>(y), data<half>(db),
							first_dim, last_dim, act);
				else
					kernel_act_bias_backward<half, 1, float> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<half>(dy), data<half>(y), data<half>(db),
							first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_act_bias_backward<float, 4> <<<gridDim1_x4, blockDim1, 0, stream>>>(data<float>(dy), data<float>(y), data<float>(db),
							first_dim, last_dim, act);
				else
					kernel_act_bias_backward<float, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<float>(dy), data<float>(y), data<float>(db),
							first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_act_bias_backward<double, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<double>(dy), data<double>(y), data<double>(db),
						first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
		batch_gemm(context, 'n', 'n', 1.0f, dy, w, beta_dx, dx);
		batch_gemm(context, 't', 'n', 1.0f, dy, x, beta_dw, dw);
	}

} /* namespace ml */
