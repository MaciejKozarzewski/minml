/*
 * moe.cu
 *
 *  Created on: Feb 6, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../vec/vec_headers.cuh"
#include "../helpers/misc.cuh"
#include "../helpers/indexers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;

//	__device__ int get_index(int i) noexcept
//	{
//		return i + i / 32;
//	}
//
//	template<typename T>
//	struct pack4
//	{
//			T x0, x1, x2, x3;
//			__device__ pack4() noexcept = default;
//			__device__ pack4(const T *ptr, int offset) noexcept
//			{
//				x0 = ptr[get_index(offset + 0)];
//				x1 = ptr[get_index(offset + 1)];
//				x2 = ptr[get_index(offset + 2)];
//				x3 = ptr[get_index(offset + 3)];
//			}
//			__device__ void store(T *ptr, int offset) const noexcept
//			{
//				ptr[get_index(offset + 0)] = x0;
//				ptr[get_index(offset + 1)] = x1;
//				ptr[get_index(offset + 2)] = x2;
//				ptr[get_index(offset + 3)] = x3;
//			}
//	};
//
//	template<typename T>
//	__device__ void swap(T &lhs, T &rhs)
//	{
//		const T tmp = lhs;
//		lhs = rhs;
//		rhs = tmp;
//	}
//
//	template<typename T>
//	__device__ void sort4(pack4<T> &values, pack4<int> &indices) noexcept
//	{
//		if (values.x0 < values.x1)
//		{
//			swap(values.x0, values.x1);
//			swap(indices.x0, indices.x1);
//		}
//		if (values.x2 < values.x3)
//		{
//			swap(values.x2, values.x3);
//			swap(indices.x2, indices.x3);
//		}
//		if (values.x0 < values.x2)
//		{
//			swap(values.x0, values.x2);
//			swap(indices.x0, indices.x2);
//		}
//		if (values.x1 < values.x3)
//		{
//			swap(values.x1, values.x3);
//			swap(indices.x1, indices.x3);
//		}
//		if (values.x1 < values.x2)
//		{
//			swap(values.x1, values.x2);
//			swap(indices.x1, indices.x2);
//		}
//	}
//
//	template<typename T>
//	__global__ void kernel_select_top_k_old(const T *input, int *indices, T *values, int top_k, int first_dim, int last_dim)
//	{
//		assert(last_dim <= 512);
//		__shared__ int workspace_idx[512 + 16];
//		__shared__ T workspace_val[512 + 16];
//
//		for (int j = threadIdx.x; j < 512; j += blockDim.x)
//			workspace_idx[get_index(j)] = j;
//		for (int j = threadIdx.x; j < 512; j += blockDim.x)
//			workspace_val[get_index(j)] = (j < last_dim) ? input[blockIdx.x * last_dim + j] : static_cast<T>(-1.0e4f);
//		__syncthreads();
//		int shift = 0;
//		for (int i = 0; i < last_dim; i += 2)
//		{
//			for (int j = 4 * threadIdx.x + shift; j < last_dim; j += 4 * blockDim.x)
//			{
//				pack4<T> val(workspace_val, j);
//				pack4<int> ind(workspace_idx, j);
//				sort4(val, ind);
//				val.store(workspace_val, j);
//				ind.store(workspace_idx, j);
//			}
//			shift = 2 - shift;
//			__syncthreads();
//		}
//
//		__syncthreads();
//		if (indices != nullptr)
//			for (int j = threadIdx.x; j < top_k; j += blockDim.x)
//				indices[blockIdx.x * top_k + j] = workspace_idx[get_index(j)];
//		if (values != nullptr)
//			for (int j = threadIdx.x; j < top_k; j += blockDim.x)
//				values[blockIdx.x * top_k + j] = workspace_val[get_index(j)];
//	}

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
		const Indexer<4> output_indexer(experts, batch_size, top_k, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices[indices_indexer.at(b, e, k)];
			const T *src = input + input_indexer.at(b, token_index, 0);
			T *dst = output + output_indexer.at(e, b, k, 0);
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
	__global__ void kernel_scale_add(T *ptr, float beta, int elements)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;
		for (int i = N * tid; i < elements; i += N * stride)
		{
			if (beta == 0.0f)
			{
				const vec<T, N> zero(static_cast<T>(0.0f));
				zero.store(ptr + i);
			}
			else
			{
				vec<T, N> tmp(ptr + i);
				tmp *= vec<T, N>(beta);
				tmp.store(ptr + i);
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
		const Indexer<4> next_indexer(experts, batch_size, top_k, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int token_index = indices[indices_indexer.at(b, e, k)];
			const T *src = gradient_next + next_indexer.at(e, b, k, 0);
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

		const Indexer<4> input_indexer(experts, batch_size, top_k, channels);
		const Indexer<3> output_indexer(batch_size, tokens, channels);
		const Indexer<3> indices_indexer(batch_size, experts, top_k);

		for (int k = threadIdx.y; k < top_k; k += blockDim.y)
		{
			const int idx = indices_indexer.at(b, e, k);
			const int token_index = indices[idx];
			const vec<T, N> scale(scales[idx]);
			const T *src = input + input_indexer.at(e, b, k, 0);
			T *dst = output + output_indexer.at(b, token_index, 0);
			for (int c = N * threadIdx.x; c < channels; c += N * blockDim.x)
				atomic_add(dst + c, vec<T, N>(src + c) * scale);
		}
	}
	template<typename T, int N>
	__global__ void kernel_scatter_backward(const T *gradient_next, float beta1, T *gradient_prev, const T *input, const int *indices,
			const T *scales, float beta2, T *gradient_scales, int batch_size, int tokens, int channels, int experts, int top_k)
	{
		assert(channels % N == 0);
		extern __shared__ char shared_array[];
		T *workspace = reinterpret_cast<T*>(shared_array);

		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		for (int i = tid; i < tokens; i += blockDim.x * blockDim.y)
			workspace[i] = static_cast<T>(0.0f);
		__syncthreads();

		const int b = blockIdx.x;
		const int e = blockIdx.y;

		const Indexer<4> prev_indexer(experts, batch_size, top_k, channels);
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
				const vec<T, N> x(input + prev_indexer.at(e, b, k, c));
				vec<T, N> dx = dy * scale;
				scale_gradient += dy * x;
				if (beta1 != 0.0f)
					dx += vec<T, N>(beta1) * vec<T, N>(gradient_prev + prev_indexer.at(e, b, k, c));
				dx.store(gradient_prev + prev_indexer.at(e, b, k, c));
			}

			T reduced_scale_gradient = horizontal_add(scale_gradient);
			for (int i = 16; i >= 1; i /= 2)
				reduced_scale_gradient += __shfl_xor_sync(0xffffffff, reduced_scale_gradient, i);
			if (threadIdx.x == 0)
				workspace[token_index] = reduced_scale_gradient;
		}

		__syncthreads();
		const Indexer<3> ds_indexer(batch_size, experts, tokens);
		for (int i = tid; i < tokens; i += blockDim.x * blockDim.y)
		{
			T tmp = workspace[i];
			if (beta2 != 0.0f)
				tmp += static_cast<T>(gradient_scales[ds_indexer.at(b, e, i)]) * static_cast<T>(beta2);
			gradient_scales[ds_indexer.at(b, e, i)] = tmp;
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
		const int experts = y.dim[0];
		assert(batch_size == y.dim[1]);
		const int top_k = y.dim[2];
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
		const int experts = dy.dim[0];
		assert(batch_size == dy.dim[1]);
		const int top_k = dy.dim[2];
		assert(channels == dy.dim[3]);
		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		const int elements = volume(dx);
		switch (dx.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (elements % 4 == 0)
					kernel_scale_add<half, 4> <<<1024, 256, 0, stream>>>(data<half>(dx), beta, volume(dx));
				else
					kernel_scale_add<half, 1> <<<1024, 256, 0, stream>>>(data<half>(dx), beta, volume(dx));
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (elements % 4 == 0)
					kernel_scale_add<float, 4> <<<1024, 256, 0, stream>>>(data<float>(dx), beta, volume(dx));
				else
					kernel_scale_add<float, 1> <<<1024, 256, 0, stream>>>(data<float>(dx), beta, volume(dx));
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_scale_add<double, 1> <<<1024, 256, 0, stream>>>(data<double>(dx), beta, volume(dx));
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);

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
		const int experts = x.dim[0];
		assert(batch_size == x.dim[1]);
		const int top_k = x.dim[2];
		assert(channels == x.dim[3]);
		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		const int elements = volume(y);
		switch (y.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (elements % 4 == 0)
					kernel_scale_add<half, 4> <<<1024, 256, 0, stream>>>(data<half>(y), beta, elements);
				else
					kernel_scale_add<half, 1> <<<1024, 256, 0, stream>>>(data<half>(y), beta, elements);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (elements % 4 == 0)
					kernel_scale_add<float, 4> <<<1024, 256, 0, stream>>>(data<float>(y), beta, elements);
				else
					kernel_scale_add<float, 1> <<<1024, 256, 0, stream>>>(data<float>(y), beta, elements);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_scale_add<double, 1> <<<1024, 256, 0, stream>>>(data<double>(y), beta, elements);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);

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
		const int experts = x.dim[0];
		assert(batch_size == x.dim[1]);
		const int top_k = x.dim[2];
		assert(channels == x.dim[3]);
		assert(batch_size == indices.dim[0]);
		assert(experts == indices.dim[1]);
		assert(top_k == indices.dim[2]);

		dim3 block_dim(32, 8);
		dim3 grid_dim(batch_size, experts);
		const size_t shared_memory_size = size_of(x.dtype) * tokens;
		switch (x.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (channels % 4 == 0)
					kernel_scatter_backward<half, 4> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(dy), beta1, data<half>(dx),
							data<half>(x), data<int>(indices), data<half>(scales), beta2, data<half>(dscales), batch_size, tokens, channels, experts,
							top_k);
				else
					kernel_scatter_backward<half, 1> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<half>(dy), beta1, data<half>(dx),
							data<half>(x), data<int>(indices), data<half>(scales), beta2, data<half>(dscales), batch_size, tokens, channels, experts,
							top_k);
				break;
			}
			case DTYPE_FLOAT32:
			{
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
				kernel_scatter_backward<double, 1> <<<grid_dim, block_dim, shared_memory_size, stream>>>(data<double>(dy), beta1, data<double>(dx),
						data<double>(x), data<int>(indices), data<double>(scales), beta2, data<double>(dscales), batch_size, tokens, channels,
						experts, top_k);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

} /* namespace ml */
