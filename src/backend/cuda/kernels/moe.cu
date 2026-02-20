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
		extern __shared__ char shared_array[];
#if __CUDA_ARCH__ >= FP16_MIN_ARCH
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
	/*
	 * indices	[N x E x K]
	 * flags	[N x T x E]		true if given token is being used by given expert within the batch, false otherwise
	 */
	__global__ void kernel_invert_indices(const int *indices, bool *flags, int batch_size, int experts, int top_k, int tokens)
	{

	}

	template<typename T, int N>
	__global__ void kernel_gather_forward(const T *input, float beta, T *output, const int *indices, int batch_size, int tokens, int channels,
			int experts, int top_k)
	{
		assert(channels % N == 0);
		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<3> input_indexer(batch_size, tokens, channels);
		const Indexer<4> output_indexer(batch_size, top_k, experts, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices[indices_indexer.at(b, e, k)];
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
	template<typename T, int N>
	__global__ void kernel_gather_backward(const T *gradient_next, T *gradient_prev, const int *indices, int batch_size, int tokens, int channels,
			int experts, int top_k)
	{
		assert(channels % N == 0);
		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<3> prev_indexer(batch_size, tokens, channels);
		const Indexer<4> next_indexer(batch_size, top_k, experts, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices[indices_indexer.at(b, e, k)];
			const T *src = gradient_next + next_indexer.at(b, k, e, 0);
			T *dst = gradient_prev + prev_indexer.at(b, token_index, 0);
			for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
				atomic_add(dst + c, vec<T, N>(src + c));
		}
	}

	template<typename T, int N>
	__global__ void kernel_scatter_forward(const T *input, T *output, const int *indices, const T *scales, int batch_size, int tokens, int channels,
			int experts, int top_k)
	{
		assert(channels % N == 0);
		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<4> input_indexer(batch_size, top_k, experts, channels);
		const Indexer<3> output_indexer(batch_size, tokens, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int idx = indices_indexer.at(b, e, k);
			const int token_index = indices[idx];
			const vec<T, N> scale(scales[idx]);
			const T *src = input + input_indexer.at(b, k, e, 0);
			T *dst = output + output_indexer.at(b, token_index, 0);
			for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
				atomic_add(dst + c, vec<T, N>(src + c) * scale);
		}
	}
	template<typename T, int N, typename U = T>
	__global__ void kernel_scatter_backward(const T *gradient_next, float beta1, T *gradient_prev, const T *input, const int *indices,
			const T *scales, float beta2, T *gradient_scales, int batch_size, int tokens, int channels, int experts, int top_k)
	{
		assert(channels % N == 0);
		extern __shared__ char shared_array[];
		U *workspace = reinterpret_cast<U*>(shared_array);

		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tid; i < tokens; i += blockDim.x * blockDim.y)
			workspace[i] = static_cast<T>(0.0f);
		__syncthreads();

		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<4> prev_indexer(batch_size, top_k, experts, channels);
		const Indexer<3> next_indexer(batch_size, tokens, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int idx = indices_indexer.at(b, e, k);
			const int token_index = indices[idx];
			const vec<T, N> scale(scales[idx]);
			vec<T, N> scale_gradient(static_cast<T>(0.0f));
			for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
			{
				const vec<T, N> dy(gradient_next + next_indexer.at(b, token_index, c));
				const vec<T, N> x(input + prev_indexer.at(b, k, e, c));
				vec<T, N> dx = dy * scale;
				scale_gradient += dy * x;
				if (beta1 != 0.0f)
					dx += vec<T, N>(beta1) * vec<T, N>(gradient_prev + prev_indexer.at(b, k, e, c));
				dx.store(gradient_prev + prev_indexer.at(b, k, e, c));
			}

			U reduced_scale_gradient = horizontal_add(scale_gradient);
			for (int i = 16; i >= 1; i /= 2)
				reduced_scale_gradient += __shfl_xor_sync(0xffffffff, reduced_scale_gradient, i);
			if (threadIdx.x == 0)
				workspace[token_index] = reduced_scale_gradient;
		}

		__syncthreads();
		const Indexer<3> ds_indexer(batch_size, experts, tokens);
		for (int i = tid; i < tokens; i += blockDim.x * blockDim.y)
		{
			U tmp = workspace[i];
			if (beta2 != 0.0f)
				tmp += static_cast<U>(gradient_scales[ds_indexer.at(b, e, i)]) * static_cast<U>(beta2);
			gradient_scales[ds_indexer.at(b, e, i)] = static_cast<T>(tmp);
		}
	}

	void print_shape(const mlTensor_t &t)
	{
		for (int i = 0; i < t.rank; i++)
			std::cout << ((i != 0) ? std::string(" x ") : std::string()) << t.dim[i];
		std::cout << '\n';
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
}

namespace ml
{
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
	void cuda_gather_tokens_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices, float beta, mlTensor_t y)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = x.dim[0];
		const int tokens = x.dim[1] * x.dim[2];
		const int channels = x.dim[3];

		assert(batch_size == y.dim[0]);
		const int top_k = y.dim[1];
		const int experts = y.dim[2];
		assert(channels == y.dim[3]);

		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);

		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_gather_forward<half, 4> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), beta, data<half>(y), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_forward<half, 1> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), beta, data<half>(y), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_gather_forward<float, 4> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), beta, data<float>(y), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_forward<float, 1> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), beta, data<float>(y), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_gather_forward<double, 1> <<<grid_dim, block_dim, 0, stream>>>(data<double>(x), beta, data<double>(y), data<int>(indices),
						batch_size, tokens, channels, experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_gather_tokens_backward(mlContext_t context, const mlTensor_t dy, const mlTensor_t indices, float beta, mlTensor_t dx)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = dx.dim[0];
		const int tokens = dx.dim[1] * dx.dim[2];
		const int channels = dx.dim[3];

		assert(batch_size == dy.dim[0]);
		const int top_k = dy.dim[1];
		const int experts = dy.dim[2];
		assert(channels == dy.dim[3]);

		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		cuda_scale_tensor(context, beta, dx);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_gather_backward<half, 4> <<<grid_dim, block_dim, 0, stream>>>(data<half>(dy), data<half>(dx), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_backward<half, 1> <<<grid_dim, block_dim, 0, stream>>>(data<half>(dy), data<half>(dx), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_gather_backward<float, 4> <<<grid_dim, block_dim, 0, stream>>>(data<float>(dy), data<float>(dx), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				else
					kernel_gather_backward<float, 1> <<<grid_dim, block_dim, 0, stream>>>(data<float>(dy), data<float>(dx), data<int>(indices),
							batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_gather_backward<double, 1> <<<grid_dim, block_dim, 0, stream>>>(data<double>(dy), data<double>(dx), data<int>(indices),
						batch_size, tokens, channels, experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_scatter_tokens_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices, const mlTensor_t scales, float beta,
			mlTensor_t y)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = y.dim[0];
		const int tokens = y.dim[1] * y.dim[2];
		const int channels = y.dim[3];

		assert(batch_size == x.dim[0]);
		const int top_k = x.dim[1];
		const int experts = x.dim[2];
		assert(channels == x.dim[3]);

		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		cuda_scale_tensor(context, beta, y);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_scatter_forward<half, 4> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), data<half>(y), data<int>(indices),
							data<half>(scales), batch_size, tokens, channels, experts, top_k);
				else
					kernel_scatter_forward<half, 1> <<<grid_dim, block_dim, 0, stream>>>(data<half>(x), data<half>(y), data<int>(indices),
							data<half>(scales), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (channels % 4 == 0)
					kernel_scatter_forward<float, 4> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), data<float>(y), data<int>(indices),
							data<float>(scales), batch_size, tokens, channels, experts, top_k);
				else
					kernel_scatter_forward<float, 1> <<<grid_dim, block_dim, 0, stream>>>(data<float>(x), data<float>(y), data<int>(indices),
							data<float>(scales), batch_size, tokens, channels, experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_scatter_forward<double, 1> <<<grid_dim, block_dim, 0, stream>>>(data<double>(x), data<double>(y), data<int>(indices),
						data<double>(scales), batch_size, tokens, channels, experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_scatter_tokens_backward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices, const mlTensor_t scales, const mlTensor_t dy,
			float beta1, mlTensor_t dx, float beta2, mlTensor_t dscales)
	{
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int batch_size = dy.dim[0];
		const int tokens = dy.dim[1] * dy.dim[2];
		const int channels = dy.dim[3];

		assert(batch_size == x.dim[0]);
		const int top_k = x.dim[1];
		const int experts = x.dim[2];
		assert(channels == x.dim[3]);

		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);
		
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				const size_t shared_memory_size = sizeof(float) * tokens;
				if (channels % 4 == 0)
					kernel_scatter_backward<half, 4, float> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(dy), beta1, data<half>(dx),
							data<half>(x), data<int>(indices), data<half>(scales), beta2, data<half>(dscales), batch_size, tokens, channels, experts,
							top_k);
				else
					kernel_scatter_backward<half, 1, float> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(dy), beta1, data<half>(dx),
							data<half>(x), data<int>(indices), data<half>(scales), beta2, data<half>(dscales), batch_size, tokens, channels, experts,
							top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
				const size_t shared_memory_size = sizeof(float) * tokens;
				if (channels % 4 == 0)
					kernel_scatter_backward<float, 4> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<float>(dy), beta1, data<float>(dx),
							data<float>(x), data<int>(indices), data<float>(scales), beta2, data<float>(dscales), batch_size, tokens, channels,
							experts, top_k);
				else
					kernel_scatter_backward<float, 1> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<float>(dy), beta1, data<float>(dx),
							data<float>(x), data<int>(indices), data<float>(scales), beta2, data<float>(dscales), batch_size, tokens, channels,
							experts, top_k);
				break;
			}
			case DTYPE_FLOAT64:
			{
				const size_t shared_memory_size = sizeof(double) * tokens;
				kernel_scatter_backward<double, 1> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<double>(dy), beta1, data<double>(dx),
						data<double>(x), data<int>(indices), data<double>(scales), beta2, data<double>(dscales), batch_size, tokens, channels,
						experts, top_k);
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
