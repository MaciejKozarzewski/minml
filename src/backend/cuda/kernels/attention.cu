/*
 * attention.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include "../vec/vec_headers.cuh"
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cinttypes>
#include <iostream>

namespace
{
	using namespace vectors;

	struct Index2D
	{
			int8_t x = 0;
			int8_t y = 0;
	};

	template<typename T>
	__host__ __device__ T clamp(T x, T lower, T upper)
	{
		assert(lower <= upper);
		return max(lower, min(upper, x));
	}
	__host__ __device__ int round_up(int x, int y)
	{
		const int tmp = x % y;
		return (tmp == 0) ? x : (x + y - tmp);
	}
	__host__ __device__ float to_float(int32_t x)
	{
		return reinterpret_cast<const float*>(&x)[0];
	}
	__host__ __device__ uint32_t to_uint32(float x)
	{
		return reinterpret_cast<const uint32_t*>(&x)[0];
	}

	__device__ void* apply_offset(void *ptr, int offsetInBytes)
	{
		return reinterpret_cast<uint8_t*>(ptr) + offsetInBytes;
	}
	__global__ void kernel_calculate_pointers(void *q_ptr[], void *k_ptr[], void *v_ptr[], void *input, void *qk_ptr[], void *workspace,
			void *out_ptr[], void *output, int batch_size, int tokens, int num_heads, int head_dim, int dtype_size, bool symmetric)
	{
		const Indexer<5> input_indexer(batch_size, tokens, 3 - symmetric, num_heads, head_dim);
		const Indexer<4> workspace_indexer(batch_size, num_heads, tokens, tokens);
		const Indexer<4> output_indexer(batch_size, tokens, num_heads, head_dim);

		for (int i = threadIdx.x; i < batch_size * num_heads; i += blockDim.x)
		{
			const int idx_b = i / num_heads;
			const int idx_h = i % num_heads;

			q_ptr[i] = apply_offset(input, dtype_size * input_indexer.at(idx_b, 0, 0, idx_h, 0));
			if (symmetric)
				k_ptr[i] = q_ptr[i];
			else
				k_ptr[i] = apply_offset(input, dtype_size * input_indexer.at(idx_b, 0, 1, idx_h, 0));
			v_ptr[i] = apply_offset(input, dtype_size * input_indexer.at(idx_b, 0, 2 - symmetric, idx_h, 0));
			qk_ptr[i] = apply_offset(workspace, dtype_size * workspace_indexer.at(idx_b, idx_h, 0, 0));
			out_ptr[i] = apply_offset(output, dtype_size * output_indexer.at(idx_b, 0, idx_h, 0));
		}
	}

	template<int N, typename T, typename U>
	__device__ void vector_copy(T *dst, const U *src)
	{
		store_vec(dst, load_vec<U, N>(src));
	}

	template<typename T, bool UseBias>
	__global__ void kernel_softmax_forward_in_place(T *input, const T *weights, int batch_size, int num_heads, int height, int width,
			int weights_size, const uint32_t *mask)
	{
		extern __shared__ char shared_array[];

		uint32_t *shared_mask = reinterpret_cast<uint32_t*>(shared_array);
		float *shared_input = reinterpret_cast<float*>(shared_mask + height * width);

		float *shared_biases = nullptr;
		Index2D *indices = nullptr;

		const int block_size = blockDim.x * blockDim.y;
		const int tid = threadIdx.y * blockDim.x + threadIdx.x;

		const int token_idx = blockDim.y * blockIdx.x;
		const int batch_idx = blockIdx.y;
		const int head_idx = blockIdx.z;
		const int tokens = height * width;

		if (UseBias)
		{
			shared_biases = reinterpret_cast<float*>(shared_input + tokens * blockDim.y);
			indices = reinterpret_cast<Index2D*>(shared_biases + weights_size * round_up(weights_size, 4));

			for (int i = tid; i < tokens; i += block_size)
			{
				indices[i].x = i / width;
				indices[i].y = i - indices[i].x * width;
			}
			const int tmp = weights_size * round_up(weights_size, 4);
			const Indexer<2> weight_indexer(num_heads, tmp);
			for (int i = 4 * tid; i < tmp; i += 4 * block_size)
				vector_copy<4>(shared_biases + i, weights + weight_indexer.at(head_idx, i));
		}

		if (mask != nullptr)
		{
			const Indexer<2> mask_indexer(batch_size, tokens);
			for (int i = tid; i < tokens; i += block_size)
				shared_mask[i] = mask[mask_indexer.at(batch_idx, i)];
		}
		else
			for (int i = tid; i < tokens; i += block_size)
				shared_mask[i] = 0xFFFFFFFFu;
		__syncthreads();

		const Indexer<4> input_indexer(batch_size, num_heads, tokens, tokens);
		const int idx = input_indexer.at(batch_idx, head_idx, token_idx, 0);
		const int tokens_left = min(tokens - token_idx, blockDim.y) * tokens;
		if (idx % 4 == 0)
		{
			for (int j = 4 * tid; j < tokens_left; j += 4 * block_size)
			{
				vec<float, 4> tmp;
				if (tokens_left - j >= 4)
					tmp = load_vec<float, 4>(input + idx + j);
				else
					tmp = partial_load_vec<float, 4>(input + idx + j, tokens_left - j);
				store_vec(shared_input + j, tmp);
			}
		}
		else
		{ // unaligned loads
			for (int j = tid; j < tokens_left; j += block_size)
				shared_input[j] = input[idx + j];
		}
		__syncthreads();

		const Index2D origin = indices[token_idx + threadIdx.y];
		const Indexer<2> weight_indexer(weights_size, round_up(weights_size, 4));
		const int range = (weights_size - 1) / 2;

		const uint32_t mask1 = shared_mask[token_idx + threadIdx.y];

		float max_value = -1e+32f;
		for (int j = threadIdx.x; j < tokens; j += blockDim.x)
		{
			float tmp = shared_input[threadIdx.y * tokens + j];
			if (UseBias)
			{
				const Index2D current = indices[j];
				const int offset_x = range + clamp(current.x - origin.x, -range, range);
				const int offset_y = range + clamp(current.y - origin.y, -range, range);
				tmp += shared_biases[weight_indexer.at(offset_x, offset_y)];
			}

			max_value = max(max_value, tmp);
			shared_input[threadIdx.y * tokens + j] = tmp;
		}
		for (int k = 16; k >= 1; k /= 2)
			max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, k));

		float partial_sum = 0.0f;
		for (int j = threadIdx.x; j < tokens; j += blockDim.x)
		{
			float tmp = exp(shared_input[threadIdx.y * tokens + j] - max_value);

			const uint32_t mask2 = shared_mask[j];
			tmp = to_float(to_uint(tmp) & (mask1 & mask2));

			partial_sum += tmp;
			shared_input[threadIdx.y * tokens + j] = tmp;
		}
		for (int k = 16; k >= 1; k /= 2)
			partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, k);
		const float inv_sum = 1.0f / partial_sum;

		for (int j = threadIdx.x; j < tokens; j += blockDim.x)
			shared_input[threadIdx.y * tokens + j] *= inv_sum;

		__syncthreads();
		if (idx % 4 == 0)
		{
			for (int j = 4 * tid; j < tokens_left; j += 4 * block_size)
			{
				const vec<float, 4> tmp = load_vec<float, 4>(shared_input + j);
				if (tokens_left - j >= 4)
					store_vec(input + idx + j, tmp);
				else
					partial_store_vec(input + idx + j, convert<T>(tmp), tokens_left - j);
			}
		}
		else
		{ // unaligned stores
			for (int j = tid; j < tokens_left; j += block_size)
				input[idx + j] = shared_input[j];
		}
	}
	template<typename T, bool UseBias>
	__global__ void kernel_softmax_backward_in_place(const T *output, T *gradient, T *weights_update, int batch_size, int num_heads, int height,
			int width, int weights_size, float *mask_update)
	{
		__shared__ cg::block_tile_memory<128> btm;
		cg::thread_block thb = cg::this_thread_block(btm);
		cg::thread_block_tile < 128 > tile = cg::tiled_partition<128>(thb);

		extern __shared__ char shared_array[];

		const int batch_idx = blockIdx.x;
		const int head_idx = blockIdx.y;
		const int tokens = height * width;

		float *shared_output = reinterpret_cast<float*>(shared_array);
		float *shared_gradient = shared_output + tokens;
		float *shared_weight_update = nullptr;
		Index2D *indices = nullptr;

		if (UseBias)
		{
			shared_weight_update = shared_gradient + tokens;
			indices = reinterpret_cast<Index2D*>(shared_weight_update + weights_size * round_up(weights_size, 4));

			for (int i = threadIdx.x; i < tokens; i += blockDim.x)
			{
				indices[i].x = i / width;
				indices[i].y = i - indices[i].x * width;
			}
			for (int i = threadIdx.x; i < weights_size * round_up(weights_size, 4); i += blockDim.x)
				shared_weight_update[i] = 0.0f;
			__syncthreads();
		}

		const Indexer<4> gradient_indexer(batch_size, num_heads, tokens, tokens);
		const Indexer<2> weight_indexer(weights_size, round_up(weights_size, 4));
		for (int i = 0; i < tokens; i++)
		{
			const int idx = gradient_indexer.at(batch_idx, head_idx, i, 0);

			float local_sum = 0.0f;
			for (int j = threadIdx.x; j < tokens; j += blockDim.x)
			{
				const float out = output[idx + j];
				const float grad = gradient[idx + j];
				local_sum += out * grad;
				shared_output[j] = out;
				shared_gradient[j] = grad;
			}
			local_sum = cg::reduce(tile, local_sum, cg::plus<float>());

			const Index2D origin = indices[i];
			const int range = (weights_size - 1) / 2;
			for (int j = threadIdx.x; j < tokens; j += blockDim.x)
			{
				const float dx = shared_output[j] * (shared_gradient[j] - local_sum);

				if (UseBias)
				{
					const Index2D current = indices[j];
					const int offset_h = range + clamp(current.x - origin.x, -range, range);
					const int offset_w = range + clamp(current.y - origin.y, -range, range);
					atomicAdd(shared_weight_update + weight_indexer.at(offset_h, offset_w), dx);
				}
				if (mask_update != nullptr)
				{
					const Indexer<4> mask_indexer(batch_size, num_heads, tokens, tokens);
					mask_update[mask_indexer.at(batch_idx, head_idx, i, j)] += dx;
				}
				gradient[idx + j] = dx;
			}
		}

		if (UseBias)
		{
			__syncthreads();
			const Indexer<3> update_indexer(batch_size, num_heads, weights_size * round_up(weights_size, 4));
			for (int i = threadIdx.x; i < weights_size * round_up(weights_size, 4); i += blockDim.x)
				weights_update[update_indexer.at(batch_idx, head_idx, i)] = shared_weight_update[i];
		}
	}
	template<typename T>
	__global__ void kernel_weights_update_reduction(const float *workspace, T *update, int batch_size, int num_heads, int last_dim)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		__shared__ float storage[32 * 32];

		const Indexer<3> workspace_indexer(batch_size, num_heads, last_dim);

		const int tid = blockIdx.x * 32 + threadIdx.x;
		float d_w = 0.0f;
		if (tid < last_dim)
			for (int i = threadIdx.y; i < batch_size; i += 32)
				d_w += workspace[workspace_indexer.at(i, blockIdx.y, tid)];
		storage[threadIdx.y * 32 + threadIdx.x] = d_w;

		__syncthreads();
		assert(blockDim.x == 32 && blockDim.y == 32);
		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
				storage[threadIdx.y * 32 + threadIdx.x] += storage[(i + threadIdx.y) * 32 + threadIdx.x];
			__syncthreads();
		}

		const Indexer<2> update_indexer(num_heads, last_dim);
		if (threadIdx.y == 0 && tid < last_dim)
			update[update_indexer.at(blockIdx.y, tid)] += storage[threadIdx.x];
	}

//	template<typename T>
//	__global__ void kernel_normalize_qk_forward(void *input[], int first_dim, int last_dim, int stride, float *workspace)
//	{
//		extern __shared__ char shared_array[];
//		float *shared_input = reinterpret_cast<float*>(shared_array);
//
//		T *input_ptr = reinterpret_cast<T*>(input[blockIdx.x]);
//
//		__shared__ cg::block_tile_memory<256> btm;
//		cg::thread_block thb = cg::this_thread_block(btm);
//		cg::thread_block_tile<32> tile = cg::tiled_partition<32>(thb);
//
//		for (int i = threadIdx.y; i < first_dim; i += blockDim.y)
//		{
//			float sum_squares = 0.0f;
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				const float in = static_cast<float>(input_ptr[i * stride + j]);
//				sum_squares += in * in;
//				shared_input[threadIdx.y * last_dim + j] = in;
//			}
//			sum_squares = cg::reduce(tile, sum_squares, cg::plus<float>());
//			const float rms = std::sqrt(sum_squares / last_dim);
//			const float inv_rms = 1.0f / (1.0e-6f + rms);
//
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				input_ptr[i * stride + j] = shared_input[threadIdx.y * last_dim + j] * inv_rms;
//
//			if (threadIdx.x == 0 and workspace != nullptr)
//				workspace[blockIdx.x * first_dim + i] = rms;
//		}
//	}
//	__global__ void kernel_normalize_qk_backward(void *gradient[], void *output[], int first_dim, int last_dim, int stride, const float *workspace)
//	{
//		assert(blockDim.x == 32);
//
//		extern __shared__ char shared_array[];
//		float *shared_input = reinterpret_cast<float*>(shared_array);
//		float *shared_gradient = shared_input + blockDim.y * last_dim;
//
//		float *output_ptr = reinterpret_cast<float*>(output[blockIdx.x]);
//		float *gradient_ptr = reinterpret_cast<float*>(gradient[blockIdx.x]);
//
//		__shared__ cg::block_tile_memory<256> btm;
//		cg::thread_block thb = cg::this_thread_block(btm);
//		cg::thread_block_tile<32> tile = cg::tiled_partition<32>(thb);
//
//		for (int i = threadIdx.y; i < first_dim; i += blockDim.y)
//		{
//			const float rms = workspace[blockIdx.x * first_dim + i];
//
//			float sum_squares = 0.0f;
//			float sum = 0.0f;
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				const float in = output_ptr[i * stride + j] * (1.0e-6f + rms);
//				const float grad = gradient_ptr[i * stride + j];
//				sum_squares += in * in;
//				sum += in * grad;
//
//				shared_input[threadIdx.y * last_dim + j] = in;
//				shared_gradient[threadIdx.y * last_dim + j] = grad;
//			}
//
//			sum_squares = cg::reduce(tile, sum_squares, cg::plus<float>());
//			sum = cg::reduce(tile, sum, cg::plus<float>());
//
//			const float mult = 1.0f / (last_dim * rms * rms * rms);
//			sum_squares *= mult;
//			sum *= mult;
//			for (int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				const float in = shared_input[threadIdx.y * last_dim + j];
//				const float grad = shared_gradient[threadIdx.y * last_dim + j];
//
//				gradient_ptr[i * stride + j] = grad * sum_squares - in * sum;
//			}
//		}
//	}

	void gemm_batched(ml::mlContext_t context, char opA, char opB, ml::mlDataType_t dtype, int M, int N, int K, float alpha, const void *A[], int lda,
			const void *B[], int ldb, float beta, void *C[], int ldc, int batch_count)
	{
		cublasOperation_t transa = ml::is_transpose(opA) ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transb = ml::is_transpose(opB) ? CUBLAS_OP_T : CUBLAS_OP_N;

		cublasHandle_t handle = ml::cuda_backend::Context::getHandle(context);
		cublasStatus_t err = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
		assert(err == CUBLAS_STATUS_SUCCESS);
		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
			{
				const half _alpha = alpha;
				const half _beta = beta;
				cublasStatus_t status = cublasHgemmBatched(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<half*>(B), ldb,
						ml::getPointer<half*>(A), lda, &_beta, ml::getPointer<half*>(C), ldc, batch_count);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
			case ml::DTYPE_FLOAT32:
			{
				const float _alpha = alpha;
				const float _beta = beta;
				if (ml::cuda_backend::Context::allowsTF32(context))
				{
					cublasStatus_t status = cublasGemmBatchedEx(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<void*>(B), CUDA_R_32F, ldb,
							ml::getPointer<void*>(A), CUDA_R_32F, lda, &_beta, ml::getPointer<void*>(C), CUDA_R_32F, ldc, batch_count,
							CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				else
				{
					cublasStatus_t status = cublasSgemmBatched(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<float*>(B), ldb,
							ml::getPointer<float*>(A), lda, &_beta, ml::getPointer<float*>(C), ldc, batch_count);
					assert(status == CUBLAS_STATUS_SUCCESS);
				}
				break;
			}
			case ml::DTYPE_FLOAT64:
			{
				const double _alpha = alpha;
				const double _beta = beta;
				cublasStatus_t status = cublasDgemmBatched(handle, transb, transa, N, M, K, &_alpha, ml::getPointer<double*>(B), ldb,
						ml::getPointer<double*>(A), lda, &_beta, ml::getPointer<double*>(C), ldc, batch_count);
				assert(status == CUBLAS_STATUS_SUCCESS);
				break;
			}
		}
	}
	void run_softmax_forward(cudaStream_t stream, void *input, ml::mlShape_t input_shape, const void *weights, ml::mlShape_t weights_shape,
			int num_heads, ml::mlDataType_t dtype, const void *mask)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const bool use_bias = weights != nullptr;
		const int range = use_bias ? weights_shape.dim[1] : 0;

		dim3 blockDim(32, 8);
		dim3 gridDim((height * width + blockDim.y - 1) / blockDim.y, batch_size, num_heads);

		int shared_mem = sizeof(float) * tokens * blockDim.y;
		if (use_bias)
		{
			assert(weights_shape.rank == 3);
			assert(weights_shape.dim[2] % 4 == 0);
			shared_mem += sizeof(float) * weights_shape.dim[1] * weights_shape.dim[2] + sizeof(Index2D) * tokens;
		}

		switch (dtype)
		{
			case ml::DTYPE_FLOAT16:
			{
				if (use_bias)
					kernel_softmax_forward_in_place<half, true> <<<gridDim, blockDim, shared_mem, stream>>>(ml::getPointer<half>(input),
							ml::getPointer<half>(weights), batch_size, num_heads, height, width, range, ml::getPointer<uint32_t>(mask));
				else
					kernel_softmax_forward_in_place<half, false> <<<gridDim, blockDim, shared_mem, stream>>>(ml::getPointer<half>(input), nullptr,
							batch_size, num_heads, height, width, 0, ml::getPointer<uint32_t>(mask));
				break;
			}
			case ml::DTYPE_FLOAT32:
			{
				if (use_bias)
					kernel_softmax_forward_in_place<float, true> <<<gridDim, blockDim, shared_mem, stream>>>(ml::getPointer<float>(input),
							ml::getPointer<float>(weights), batch_size, num_heads, height, width, range, ml::getPointer<uint32_t>(mask));
				else
					kernel_softmax_forward_in_place<float, false> <<<gridDim, blockDim, shared_mem, stream>>>(ml::getPointer<float>(input), nullptr,
							batch_size, num_heads, height, width, 0, ml::getPointer<uint32_t>(mask));
				break;
			}
			case ml::DTYPE_FLOAT64:
			{
//				blockDim = dim3(32, 4);
//				gridDim = dim3((height * width + blockDim.y - 1) / blockDim.y, batch_size, num_heads);
//				const int shared_mem = sizeof(double) * (tokens * blockDim.y + weights_shape.dim[1] * weights_shape.dim[2])
//						+ sizeof(Index2D) * tokens;
//				kernel_softmax_forward_in_place<<<gridDim, blockDim, shared_mem, stream>>>(ml::getPointer<double>(input),
//						ml::getPointer<double>(weights), batch_size, num_heads, height, width, range);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
}

namespace ml
{
	int cuda_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, int num_heads, bool training)
	{
		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int tokens = input_shape.dim[1] * input_shape.dim[2];

		int result = batch_size * num_heads * tokens * tokens;
		if (training)
		{
			result = result * 2;
			if (weights_shape.rank == 3)
				result += batch_size * num_heads * weights_shape.dim[1] * weights_shape.dim[2];
		}
		return result;
	}
	void cuda_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlShape_t bias_shape,
			mlDataType_t dtype, const void *input, void *output, const void *weights, const void *bias, const void *mask, void *workspace,
			void *backward_data, int num_heads, bool symmetric)
	{
		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / (3 - symmetric);
		const int head_dim = embedding / num_heads;
		const int qkv_stride = input_shape.dim[3];

		const int num_pointers = batch_size * num_heads;
		void **pointers = getPointer<void*>(ml::cuda_backend::Context::getWorkspace(context));

		void **q_ptr = pointers + 0 * num_pointers;
		void **k_ptr = pointers + 1 * num_pointers;
		void **v_ptr = pointers + 2 * num_pointers;
		void **qk_ptr = pointers + 3 * num_pointers;
		void **out_ptr = pointers + 4 * num_pointers;

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		void *qk_tensor_ptr = (backward_data == nullptr) ? workspace : backward_data;

		kernel_calculate_pointers<<<1, 1024, 0, stream>>>(q_ptr, k_ptr, v_ptr, const_cast<void*>(input), qk_ptr, qk_tensor_ptr, out_ptr, output,
				batch_size, tokens, num_heads, head_dim, size_of(dtype), symmetric);
		assert(cudaGetLastError() == cudaSuccess);

//		{
//			dim3 blockDim(32, 8);
//			dim3 gridDim(num_pointers);
//			const int shared_mem = size_of(dtype) * blockDim.y * head_dim;
//
//			switch (dtype)
//			{
//				case DTYPE_FLOAT32:
//					kernel_normalize_qk_forward<float> <<<gridDim, blockDim, shared_mem, stream>>>(q_ptr, tokens, head_dim, qkv_stride, nullptr);
//					if (not symmetric)
//						kernel_normalize_qk_forward<float> <<<gridDim, blockDim, shared_mem, stream>>>(k_ptr, tokens, head_dim, qkv_stride, nullptr);
//					break;
//				case DTYPE_FLOAT16:
//					kernel_normalize_qk_forward<half> <<<gridDim, blockDim, shared_mem, stream>>>(q_ptr, tokens, head_dim, qkv_stride, nullptr);
//					if (not symmetric)
//						kernel_normalize_qk_forward<half> <<<gridDim, blockDim, shared_mem, stream>>>(k_ptr, tokens, head_dim, qkv_stride, nullptr);
//					break;
//			}
//		}

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', dtype, tokens, tokens, head_dim, scale, const_cast<const void**>(q_ptr), qkv_stride,
				const_cast<const void**>(k_ptr), qkv_stride, 0.0f, qk_ptr, tokens, num_pointers);

		run_softmax_forward(stream, qk_tensor_ptr, input_shape, weights, weights_shape, num_heads, dtype, mask);

		gemm_batched(context, 'n', 'n', dtype, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(qk_ptr), tokens,
				const_cast<const void**>(v_ptr), qkv_stride, 0.0f, out_ptr, embedding, num_pointers);
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlShape_t bias_shape,
			const void *input, const void *weights, const void *bias, const void *mask, void *gradient_prev, void *gradient_next,
			void *weights_update, void *bias_update, void *mask_update, void *workspace, void *backward_data, int num_heads, bool symmetric)
	{
		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / (3 - symmetric);
		const int head_dim = embedding / num_heads;
		const int qkv_stride = input_shape.dim[3];

		const int offset = batch_size * num_heads * tokens * tokens * size_of(DTYPE_FLOAT32);
		void *qk_tensor_ptr = (backward_data == nullptr) ? workspace : backward_data;
		void *backward_workspace = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace) + offset);
		void *update_workspace = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace) + 2 * offset);

		const int num_pointers = batch_size * num_heads;
		void **pointers = getPointer<void*>(ml::cuda_backend::Context::getWorkspace(context));

		void **q_ptr = pointers + 0 * num_pointers;
		void **k_ptr = pointers + 1 * num_pointers;
		void **v_ptr = pointers + 2 * num_pointers;
		void **qk_ptr = pointers + 3 * num_pointers;
		void **out_ptr = pointers + 4 * num_pointers;

		void **dq_ptr = pointers + 5 * num_pointers;
		void **dk_ptr = pointers + 6 * num_pointers;
		void **dv_ptr = pointers + 7 * num_pointers;
		void **dqk_ptr = pointers + 8 * num_pointers;
		void **dout_ptr = pointers + 9 * num_pointers;

//		float *q_rms_workspace = reinterpret_cast<float*>(pointers + 10 * num_pointers);
//		float *k_rms_workspace = q_rms_workspace + num_pointers * tokens;

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		const float scale = 1.0f / std::sqrt(head_dim);
		kernel_calculate_pointers<<<1, 1024, 0, stream>>>(q_ptr, k_ptr, v_ptr, const_cast<void*>(input), qk_ptr, qk_tensor_ptr, out_ptr, nullptr,
				batch_size, tokens, num_heads, head_dim, size_of(DTYPE_FLOAT32), symmetric);
		kernel_calculate_pointers<<<1, 1024, 0, stream>>>(dq_ptr, dk_ptr, dv_ptr, gradient_prev, dqk_ptr, backward_workspace, dout_ptr, gradient_next,
				batch_size, tokens, num_heads, head_dim, size_of(DTYPE_FLOAT32), symmetric);

//		{
//			dim3 blockDim(32, 8);
//			dim3 gridDim(num_pointers);
//			const int shared_mem = sizeof(float) * blockDim.y * head_dim;
//
//			kernel_normalize_qk_forward<float> <<<gridDim, blockDim, shared_mem, stream>>>(q_ptr, tokens, head_dim, qkv_stride, q_rms_workspace);
//			if (not symmetric)
//				kernel_normalize_qk_forward<float> <<<gridDim, blockDim, shared_mem, stream>>>(k_ptr, tokens, head_dim, qkv_stride, k_rms_workspace);
//		}

		if (backward_data == nullptr)
		{
			gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, scale, const_cast<const void**>(q_ptr), qkv_stride,
					const_cast<const void**>(k_ptr), qkv_stride, 0.0f, qk_ptr, tokens, num_pointers);
			run_softmax_forward(stream, qk_tensor_ptr, input_shape, weights, weights_shape, num_heads, DTYPE_FLOAT32, mask);
		}

		// dqk = dy * V^T
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, 1.0f, const_cast<const void**>(dout_ptr), embedding,
				const_cast<const void**>(v_ptr), qkv_stride, 0.0f, dqk_ptr, tokens, num_pointers);
		// dV = qk^T * dy
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, const_cast<const void**>(qk_ptr), tokens,
				const_cast<const void**>(dout_ptr), embedding, 0.0f, dv_ptr, qkv_stride, num_pointers);

		dim3 blockDim(128);
		dim3 gridDim(batch_size, num_heads);
		if (weights != nullptr)
		{
			const int shared_mem = sizeof(float) * (2 * tokens + weights_shape.dim[1] * weights_shape.dim[2] + 4) + sizeof(Index2D) * tokens;
			kernel_softmax_backward_in_place<float, true> <<<gridDim, blockDim, shared_mem, stream>>>(getPointer<float>(qk_tensor_ptr),
					getPointer<float>(backward_workspace), getPointer<float>(update_workspace), batch_size, num_heads, height, width,
					weights_shape.dim[1], getPointer<float>(mask_update));
			assert(cudaGetLastError() == cudaSuccess);

			const int last_dim = weights_shape.dim[1] * weights_shape.dim[2];
			blockDim = dim3(32, 32);
			gridDim = dim3((last_dim + 31) / 32, num_heads);
			kernel_weights_update_reduction<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(update_workspace), getPointer<float>(weights_update),
					batch_size, num_heads, last_dim);
			assert(cudaGetLastError() == cudaSuccess);
		}
		else
		{ // not using bias
			const int shared_mem = sizeof(float) * (2 * tokens + 4);
			kernel_softmax_backward_in_place<float, false> <<<gridDim, blockDim, shared_mem, stream>>>(getPointer<float>(qk_tensor_ptr),
					getPointer<float>(backward_workspace), nullptr, batch_size, num_heads, height, width, 0, getPointer<float>(mask_update));
			assert(cudaGetLastError() == cudaSuccess);
		}

		// dQ = dqk * K
		gemm_batched(context, 'n', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, scale, const_cast<const void**>(dqk_ptr), tokens,
				const_cast<const void**>(k_ptr), qkv_stride, 0.0f, dq_ptr, qkv_stride, num_pointers);

		const float beta = symmetric ? 1.0f : 0.0f;
		// dK = dqk^T * Q
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, scale, const_cast<const void**>(dqk_ptr), tokens,
				const_cast<const void**>(q_ptr), qkv_stride, beta, dk_ptr, qkv_stride, num_pointers);

//		{
//			dim3 blockDim(32, 8);
//			dim3 gridDim(num_pointers);
//			const int shared_mem = sizeof(float) * blockDim.y * head_dim * 2;
//
//			kernel_normalize_qk_backward<<<gridDim, blockDim, shared_mem, stream>>>(dq_ptr, q_ptr, tokens, head_dim, qkv_stride, q_rms_workspace);
//			if (not symmetric)
//				kernel_normalize_qk_backward<<<gridDim, blockDim, shared_mem, stream>>>(dk_ptr, k_ptr, tokens, head_dim, qkv_stride, k_rms_workspace);
//		}
	}


} /* namespace ml */

