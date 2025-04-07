/*
 * tensor_op.cu
 *
 *  Created on: Nov 8, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/indexers.cuh"
#include "../vec/vec_headers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>

namespace
{
	using namespace vectors;

	template<typename ComputeT, int N, int Step, typename DstT, typename SrcT>
	__global__ void kernel_sum_over_first_dim(float beta, DstT *dst, float alpha, const SrcT *src, int first_dim, int last_dim)
	{
		assert(last_dim % N == 0);
		__shared__ ComputeT workspace[32][32 * N + 1];

		const int last_dim_idx = N * (32 * blockIdx.x + threadIdx.x);
		vec<ComputeT, N> local_sum(0.0f);
		if (last_dim_idx < last_dim)
		{
			for (int i = 32 * blockIdx.y + threadIdx.y; i < first_dim; i += 32 * gridDim.y)
			{
				const vec<SrcT, N> tmp(src + i * last_dim + last_dim_idx);
				local_sum += convert<ComputeT>(tmp);
			}
			for (int n = 0; n < N; n++)
				workspace[threadIdx.y][N * threadIdx.x + n] = local_sum[n];
		}
		__syncthreads();
		for (int n = 0; n < N; n++)
			local_sum[n] = workspace[threadIdx.x][N * threadIdx.y + n];

		for (int k = 16; k >= 1; k /= 2)
		{
			for (int n = 0; n < N; n++)
				local_sum[n] += __shfl_xor_sync(0xffffffff, local_sum[n], k);
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			for (int n = 0; n < N; n++)
				workspace[0][N * threadIdx.y + n] = local_sum[n];
		}
		__syncthreads();

		if (threadIdx.y == 0 && last_dim_idx < last_dim)
		{
			vec<ComputeT, N> tmp;
			for (int n = 0; n < N; n++)
				tmp[n] = workspace[0][N * threadIdx.x + n];
			if (Step == 1) // write to temporary storage array
			{
				const int idx = blockIdx.y * last_dim + last_dim_idx;
				convert<DstT>(tmp).store(dst + idx);
			}
			if (Step == 2) // write to final destination
			{
				tmp *= vec<ComputeT, N>(alpha);
				if (beta != 0.0f)
				{
					const vec<DstT, N> y(dst + last_dim_idx);
					tmp += vec<ComputeT, N>(beta) * convert<ComputeT>(y);
				}
				convert<DstT>(tmp).store(dst + last_dim_idx);
			}
		}
	}
	template<typename T, int N>
	__global__ void kernel_multiply_tensors(T *dst, const T *src0, const T *src1, int elements)
	{
		assert(elements % N == 0);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * N; i < elements; i += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> x0(src0 + i);
			const vec<T, N> x1(src1 + i);
			const vec<T, N> y = x0 * x1;
			y.store(dst + i);
		}
	}

	template<typename T, int N>
	__global__ void kernel_add_tensors(T *dst, float alpha1, const T *src0, float alpha2, const T *src1, int elements)
	{
		assert(elements % N == 0);
		const vec<T, N> a1(alpha1);
		const vec<T, N> a2(alpha2);
		for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * N; i < elements; i += gridDim.x * blockDim.x * N)
		{
			const vec<T, N> x0(src0 + i);
			const vec<T, N> x1(src1 + i);
			const vec<T, N> y = a1 * x0 + a2 * x1;
			y.store(dst + i);
		}
	}

	template<typename T>
	__global__ void kernel_window_partition(T *output, const T *input, int batch_size, int height, int width, int channels, int2 window_size,
			int2 offset)
	{
		const int num_windows_h = (height + window_size.x - 1) / window_size.x;
		const int num_windows_w = (width + window_size.y - 1) / window_size.y;

		const int b = blockIdx.x;
		const int h = blockIdx.y;
		const int w = blockIdx.z;

		const int x = (h + offset.x + height) % height;
		const int y = (w + offset.y + width) % width;

		const int window_idx_h = x / window_size.x;
		const int window_idx_w = y / window_size.y;

		const int idx_h = x % window_size.x;
		const int idx_w = y % window_size.y;

		const Indexer<4> input_indexer(batch_size, height, width, channels);
		const Indexer<6> output_indexer(batch_size, num_windows_h, num_windows_w, window_size.x, window_size.y, channels);

		for (int c = threadIdx.x; c < channels; c += blockDim.x)
			output[output_indexer.at(b, window_idx_h, window_idx_w, idx_h, idx_w, c)] = input[input_indexer.at(b, h, w, c)];
	}
	template<typename T>
	__global__ void kernel_window_merging(T *output, const T *input, int batch_size, int height, int width, int channels, int2 window_size,
			int2 offset)
	{
		const int num_windows_h = (height + window_size.x - 1) / window_size.x;
		const int num_windows_w = (width + window_size.y - 1) / window_size.y;

		const int b = blockIdx.x;
		const int h = blockIdx.y;
		const int w = blockIdx.z;

		const int x = (h + offset.x + height) % height;
		const int y = (w + offset.y + width) % width;

		const int window_idx_h = x / window_size.x;
		const int window_idx_w = y / window_size.y;

		const int idx_h = x % window_size.x;
		const int idx_w = y % window_size.y;

		const Indexer<6> input_indexer(batch_size, num_windows_h, num_windows_w, window_size.x, window_size.y, channels);
		const Indexer<4> output_indexer(batch_size, height, width, channels);

		for (int c = threadIdx.x; c < channels; c += blockDim.x)
			output[output_indexer.at(b, h, w, c)] = input[input_indexer.at(b, window_idx_h, window_idx_w, idx_h, idx_w, c)];
	}
	template<typename T>
	__global__ void kernel_transpose(T *dst, const T *src, int4 src_shape, int4 ordering, int rank, int volume)
	{
		__shared__ int shape[4];
		__shared__ int order[4];

		__shared__ int src_stride[4];
		__shared__ int dst_stride[4];
		if (threadIdx.x == 0)
		{
			shape[0] = src_shape.x;
			shape[1] = src_shape.y;
			shape[2] = src_shape.z;
			shape[3] = src_shape.w;

			order[0] = ordering.x;
			order[1] = ordering.y;
			order[2] = ordering.z;
			order[3] = ordering.w;
			int tmp_src = 1, tmp_dst = 1;
			for (int i = rank - 1; i >= 0; i--)
			{
				src_stride[i] = tmp_src;
				dst_stride[order[i]] = tmp_dst;
				tmp_src *= shape[i];
				tmp_dst *= shape[order[i]];
			}
		}
		__syncthreads();

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < volume; i += gridDim.x * blockDim.x)
		{
			int tmp = i, dst_idx = 0;
			for (int j = 0; j < rank; j++)
			{
				int idx = tmp / src_stride[j];
				dst_idx += idx * dst_stride[j];
				tmp -= idx * src_stride[j];
			}
			dst[dst_idx] = src[i];
		}
	}

	template<typename T, int N>
	__global__ void kernel_act_bias_copy_backward(T *gradient_next, const T *output, float beta_prev, T *gradient_prev, float *bias_gradient,
			int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 8);
		assert(last_dim % N == 0);
		__shared__ float workspace[8 * 32 * N];

		const int last_dim_idx = N * (blockDim.x * blockIdx.x + threadIdx.x);
		vec<float, N> local_sum(0.0f);
		if (last_dim_idx < last_dim)
		{
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				const int tmp_idx = i * last_dim + last_dim_idx;
				vec<float, N> dy = load_vec<float, N>(gradient_next + tmp_idx);
				if (act != ml::ACTIVATION_LINEAR)
				{
					const vec<float, N> y = load_vec<float, N>(output + tmp_idx);
					switch (act)
					{
						case ml::ACTIVATION_SIGMOID:
							dy *= y * (one<float, N>() - y);
							break;
						case ml::ACTIVATION_TANH:
							dy *= one<float, N>() - square(y);
							break;
						case ml::ACTIVATION_RELU:
							dy = select(y > zero<float, N>(), dy, zero<float, N>());
							break;
						case ml::ACTIVATION_LEAKY_RELU:
							dy = select(y > zero<float, N>(), dy, dy * vec<float, N>(0.1f));
							break;
					}
					store_vec(gradient_next + tmp_idx, dy);
				}
				local_sum += dy;
				if (gradient_prev != nullptr)
				{
					if (beta_prev != 0.0f)
						dy += vec<float, N>(beta_prev) * load_vec<float, N>(gradient_prev + tmp_idx);
					store_vec(gradient_prev + tmp_idx, dy);
				}
			}
		}

		if (bias_gradient != nullptr)
		{
			const Indexer<3> idx(blockDim.y, blockDim.x, N);
			for (int n = 0; n < N; n++)
				workspace[idx.at(threadIdx.y, threadIdx.x, n)] = local_sum[n];
			__syncthreads();

			for (int i = blockDim.y / 2; i >= 1; i /= 2)
			{
				if (threadIdx.y < i)
				{
					for (int n = 0; n < N; n++)
						workspace[idx.at(threadIdx.y, threadIdx.x, n)] += workspace[idx.at(i + threadIdx.y, threadIdx.x, n)];
				}
				__syncthreads();
			}

			if (threadIdx.y == 0 && last_dim_idx < last_dim)
			{
				for (int n = 0; n < N; n++)
					bias_gradient[blockIdx.y * last_dim + last_dim_idx + n] = workspace[idx.at(0, threadIdx.x, n)];
			}
		}
	}

	template<typename T, int N, typename U>
	void dispatch_sum_over_first_dim(ml::mlContext_t context, float alpha, const ml::mlTensor_t &src, float beta, ml::mlTensor_t &dst)
	{
		const int last_dim = volume(dst);
		const int first_dim = volume(src) / last_dim;

		assert(ml::cuda::Context::getWorkspaceSize(context) >= last_dim * sizeof(T));

		T *workspace = ml::cuda::Context::getWorkspace<T>(context);
		const int workspace_first_dim = std::min((size_t) 256, ml::cuda::Context::getWorkspaceSize(context) / (sizeof(T) * last_dim));

		dim3 blockDim(32, 32);
		cudaStream_t stream = ml::cuda::Context::getStream(context);

		dim3 gridDim1((last_dim + 32 * N - 1) / (32 * N), workspace_first_dim);
		dim3 gridDim2((last_dim + 32 * N - 1) / (32 * N));
		kernel_sum_over_first_dim<T, N, 1> <<<gridDim1, blockDim, 0, stream>>>(0.0f, workspace, 1.0f, ml::data<U>(src), first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
		kernel_sum_over_first_dim<T, N, 2> <<<gridDim2, blockDim, 0, stream>>>(beta, ml::data<U>(dst), alpha, workspace, workspace_first_dim,
				last_dim);
		assert(cudaGetLastError() == cudaSuccess);
	}
}

namespace ml
{
	void cuda_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_multiply_tensors<half, 1> <<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), getPointer<half>(src1),
						getPointer<half>(src2), length);
				break;
			case DTYPE_FLOAT32:
				kernel_multiply_tensors<float, 1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), getPointer<float>(src1),
						getPointer<float>(src2), length);
				break;
			case DTYPE_FLOAT64:
				kernel_multiply_tensors<double, 1> <<<gridDim, blockDim, 0, stream>>>(getPointer<double>(dst), getPointer<double>(src1),
						getPointer<double>(src2), length);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, float beta, void *dst, float alpha1, const void *src1,
			float alpha2, const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = cuda::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = cuda::Context::getStream(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (length % 8 == 0)
					kernel_add_tensors<half, 8> <<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), alpha1, getPointer<half>(src1), alpha2,
							getPointer<half>(src2), length);
				else
					kernel_add_tensors<half, 1> <<<gridDim, blockDim, 0, stream>>>(getPointer<half>(dst), alpha1, getPointer<half>(src1), alpha2,
							getPointer<half>(src2), length);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (length % 4 == 0)
					kernel_add_tensors<float, 4> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), alpha1, getPointer<float>(src1), alpha2,
							getPointer<float>(src2), length);
				else
					kernel_add_tensors<float, 1> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dst), alpha1, getPointer<float>(src1), alpha2,
							getPointer<float>(src2), length);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_add_tensors<double, 1> <<<gridDim, blockDim, 0, stream>>>(getPointer<double>(dst), alpha1, getPointer<double>(src1), alpha2,
						getPointer<double>(src2), length);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_sum_over_first_dim(mlContext_t context, float alpha, const mlTensor_t src, float beta, mlTensor_t dst)
	{
		const int last_dim = volume(dst);
		switch (src.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					dispatch_sum_over_first_dim<float, 4, half>(context, alpha, src, beta, dst);
				else
					dispatch_sum_over_first_dim<float, 1, half>(context, alpha, src, beta, dst);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					dispatch_sum_over_first_dim<float, 4, float>(context, alpha, src, beta, dst);
				else
					dispatch_sum_over_first_dim<float, 1, float>(context, alpha, src, beta, dst);
				break;
			}
			case DTYPE_FLOAT64:
			{
				dispatch_sum_over_first_dim<double, 1, double>(context, alpha, src, beta, dst);
				break;
			}
			default:
				break;
		}
	}

	void cuda_window_partitioning(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, const void *input,
			void *output, mlShape_t offset)
	{
		cudaStream_t stream = cuda::Context::getStream(context);

		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int channels = input_shape.dim[3];

		dim3 blockDim(std::min(128, channels));
		dim3 gridDim(batch_size, height, width);

		const int2 window_size { output_shape.dim[1], output_shape.dim[2] };
		const int2 window_offset { offset.dim[0], offset.dim[1] };

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_window_partition<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT32:
				kernel_window_partition<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT64:
				kernel_window_partition<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), getPointer<double>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_window_merging(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, const void *input, void *output,
			mlShape_t offset)
	{
		cudaStream_t stream = cuda::Context::getStream(context);

		const int batch_size = output_shape.dim[0];
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int channels = output_shape.dim[3];

		dim3 blockDim(std::min(128, channels));
		dim3 gridDim(batch_size, height, width);

		const int2 window_size { input_shape.dim[1], input_shape.dim[2] };
		const int2 window_offset { offset.dim[0], offset.dim[1] };

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_window_merging<<<gridDim, blockDim, 0, stream>>>(getPointer<half>(output), getPointer<half>(input), batch_size, height, width,
						channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT32:
				kernel_window_merging<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(output), getPointer<float>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
			case DTYPE_FLOAT64:
				kernel_window_merging<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(output), getPointer<double>(input), batch_size, height,
						width, channels, window_size, window_offset);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_transpose(mlContext_t context, mlDataType_t dtype, mlShape_t dst_shape, mlShape_t src_shape, void *dst, const void *src,
			const int *ordering)
	{
		assert(volume(src_shape) == volume(dst_shape));
		assert(src_shape.rank == dst_shape.rank);

		const int4 shape { src_shape.dim[0], src_shape.dim[1], src_shape.dim[2], src_shape.dim[3] };
		const int4 order { ordering[0], ordering[1], ordering[2], ordering[3] };

		const int elements = volume(src_shape);
		const int rank = src_shape.rank;
		dim3 blockDim(256);
		dim3 gridDim(512);

		cudaStream_t stream = cuda::Context::getStream(context);
		switch (size_of(dtype))
		{
			case 1:
				kernel_transpose<<<gridDim, blockDim, 0, stream>>>(getPointer<int8_t>(dst), getPointer<int8_t>(src), shape, order, rank, elements);
				break;
			case 2:
				kernel_transpose<<<gridDim, blockDim, 0, stream>>>(getPointer<int16_t>(dst), getPointer<int16_t>(src), shape, order, rank, elements);
				break;
			case 4:
				kernel_transpose<<<gridDim, blockDim, 0, stream>>>(getPointer<int32_t>(dst), getPointer<int32_t>(src), shape, order, rank, elements);
				break;
			case 8:
				kernel_transpose<<<gridDim, blockDim, 0, stream>>>(getPointer<int2>(dst), getPointer<int2>(src), shape, order, rank, elements);
				break;
			case 16:
				kernel_transpose<<<gridDim, blockDim, 0, stream>>>(getPointer<int4>(dst), getPointer<int4>(src), shape, order, rank, elements);
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_fused_act_bias_copy_backward(mlContext_t context, mlTensor_t dy, const mlTensor_t y, float beta_dx, mlTensor_t dx, float beta_dw,
			mlTensor_t dw, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(y);
		const int last_dim = get_last_dim(y);

		assert(ml::cuda::Context::getWorkspaceSize(context) >= last_dim * sizeof(float));

		cudaStream_t stream = ml::cuda::Context::getStream(context);
		float *workspace = ml::cuda::Context::getWorkspace<float>(context);
		const int workspace_first_dim = std::min((size_t) 512, ml::cuda::Context::getWorkspaceSize(context) / (sizeof(float) * last_dim));

		dim3 blockDim1(32, 8);
		dim3 gridDim1_x1((last_dim + 31) / 32, workspace_first_dim);
		dim3 gridDim1_x4((last_dim + 127) / 127, workspace_first_dim);

		float *dw_tmp_ptr = (dw.data == nullptr) ? nullptr : workspace;

		switch (y.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_act_bias_copy_backward<half, 4> <<<gridDim1_x4, blockDim1, 0, stream>>>(data<half>(dy), data<half>(y), beta_dx,
							data<half>(dx), dw_tmp_ptr, first_dim, last_dim, act);
				else
					kernel_act_bias_copy_backward<half, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<half>(dy), data<half>(y), beta_dx,
							data<half>(dx), dw_tmp_ptr, first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_act_bias_copy_backward<float, 4> <<<gridDim1_x4, blockDim1, 0, stream>>>(data<float>(dy), data<float>(y), beta_dx,
							data<float>(dx), dw_tmp_ptr, first_dim, last_dim, act);
				else
					kernel_act_bias_copy_backward<float, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<float>(dy), data<float>(y), beta_dx,
							data<float>(dx), dw_tmp_ptr, first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

		dim3 blockDim2(32, 32);
		dim3 gridDim2_x1((last_dim + 31) / 32);
		dim3 gridDim2_x4((last_dim + 127) / 127);
		if (dw.data != nullptr)
		{
			switch (dw.dtype)
			{
				case DTYPE_FLOAT16:
				{
					if (last_dim % 4 == 0)
						kernel_sum_over_first_dim<float, 4, 2> <<<gridDim2_x4, blockDim2, 0, stream>>>(beta_dw, data<half>(dw), 1.0f, dw_tmp_ptr,
								workspace_first_dim, last_dim);
					else
						kernel_sum_over_first_dim<float, 1, 2> <<<gridDim2_x1, blockDim2, 0, stream>>>(beta_dw, data<half>(dw), 1.0f, dw_tmp_ptr,
								workspace_first_dim, last_dim);
					break;
				}
				case DTYPE_FLOAT32:
				{
					if (last_dim % 4 == 0)
						kernel_sum_over_first_dim<float, 4, 2> <<<gridDim2_x4, blockDim2, 0, stream>>>(beta_dw, data<float>(dw), 1.0f, dw_tmp_ptr,
								workspace_first_dim, last_dim);
					else
						kernel_sum_over_first_dim<float, 1, 2> <<<gridDim2_x1, blockDim2, 0, stream>>>(beta_dw, data<float>(dw), 1.0f, dw_tmp_ptr,
								workspace_first_dim, last_dim);
					break;
				}
				default:
					break;
			}
			assert(cudaGetLastError() == cudaSuccess);
		}
	}
} /* namespace ml */
