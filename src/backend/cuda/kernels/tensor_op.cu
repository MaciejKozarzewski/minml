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

	template<typename T, int N, typename U = T>
	__global__ void kernel_act_bias_copy_backward(T *gradient_next, const T *output, float beta_prev, T *gradient_prev, T *bias_gradient,
			int first_dim, int last_dim, ml::mlActivationType_t act)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 8);
		assert(last_dim % N == 0);
		__shared__ vec<U, N> workspace[8 * 32];

		const int group_idx = blockIdx.z;

		const Indexer<3> in_out_indexer(gridDim.z, first_dim, last_dim);

		const int last_dim_idx = N * (blockDim.x * blockIdx.x + threadIdx.x);
		vec<U, N> local_sum(0.0f);
		if (last_dim_idx < last_dim)
		{
			for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				const int tmp_idx = in_out_indexer.at(group_idx, i, last_dim_idx);
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
				if (gradient_prev != nullptr)
				{
					if (beta_prev != 0.0f)
						dy += vec<U, N>(beta_prev) * load_vec<U, N>(gradient_prev + tmp_idx);
					store_vec(gradient_prev + tmp_idx, dy);
				}
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
				atomic_add(bias_gradient + db_indexer.at(group_idx, last_dim_idx), tmp);
			}
		}
	}

	template<typename T, int N, typename U>
	void dispatch_sum_over_first_dim(ml::mlContext_t context, float alpha, const ml::mlTensor_t &src, float beta, ml::mlTensor_t &dst)
	{
		const int last_dim = volume(dst);
		const int first_dim = volume(src) / last_dim;

		assert(ml::cuda_backend::Context::getWorkspaceSize(context) >= last_dim * sizeof(T));

		T *workspace = ml::cuda_backend::Context::getWorkspace<T>(context);
		const int workspace_first_dim = std::min((size_t) 256, ml::cuda_backend::Context::getWorkspaceSize(context) / (sizeof(T) * last_dim));

		dim3 blockDim(32, 32);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		dim3 gridDim1((last_dim + 32 * N - 1) / (32 * N), workspace_first_dim);
		dim3 gridDim2((last_dim + 32 * N - 1) / (32 * N));
		kernel_sum_over_first_dim<T, N, 1> <<<gridDim1, blockDim, 0, stream>>>(0.0f, workspace, 1.0f, ml::data<U>(src), first_dim, last_dim);
		assert(cudaGetLastError() == cudaSuccess);
		kernel_sum_over_first_dim<T, N, 2> <<<gridDim2, blockDim, 0, stream>>>(beta, ml::data<U>(dst), alpha, workspace, workspace_first_dim,
				last_dim);
		assert(cudaGetLastError() == cudaSuccess);
	}

	template<typename T, int N>
	__global__ void kernel_axpy3(float alpha1, const T *x1, float alpha2, const T *x2, float alpha3, const T *x3, float beta, T *y, int elements)
	{
		assert(elements % N == 0);
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		const int stride = gridDim.x * blockDim.x;

		const bool load_input1 = (alpha1 != 0.0f) && (x1 != nullptr);
		const bool load_input2 = (alpha2 != 0.0f) && (x2 != nullptr);
		const bool load_input3 = (alpha3 != 0.0f) && (x3 != nullptr);
		const bool load_output = beta != 0.0f;

		const vec<T, N> zero(0.0f);
		for (int i = N * tid; i < elements; i += N * stride)
		{
			const vec<T, N> input1 = load_input1 ? vec<T, N>(x1 + i) : zero;
			const vec<T, N> input2 = load_input2 ? vec<T, N>(x2 + i) : zero;
			const vec<T, N> input3 = load_input3 ? vec<T, N>(x3 + i) : zero;
			const vec<T, N> output = load_output ? vec<T, N>(y + i) : zero;

			const vec<T, N> tmp = vec<T, N>(alpha1) * input1 + vec<T, N>(alpha2) * input2 + vec<T, N>(alpha3) * input3 + vec<T, N>(beta) * output;
			tmp.store(y + i);
		}
	}
}

namespace ml
{
	// computes x *= alpha
	void cuda_scale_tensor(mlContext_t context, float alpha, mlTensor_t x)
	{
		cuda_add_tensors(context, 0.0f, mlTensor_t(), 0.0f, mlTensor_t(), 0.0f, mlTensor_t(), alpha, x);
	}
	// computes y = alpha1 * x1 + alpha2 * x2 + alpha3 * x3 + beta * y
	void cuda_add_tensors(mlContext_t context, float alpha1, const mlTensor_t x1, float alpha2, const mlTensor_t x2, float alpha3,
			const mlTensor_t x3, float beta, mlTensor_t y)
	{
		const int elements = volume(y);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

		switch (y.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (elements % 8 == 0)
					kernel_axpy3<half, 8> <<<gridDim, blockDim, 0, stream>>>(alpha1, data<half>(x1), alpha2, data<half>(x2), alpha3, data<half>(x3),
							beta, data<half>(y), elements);
				else
					kernel_axpy3<half, 1> <<<gridDim, blockDim, 0, stream>>>(alpha1, data<half>(x1), alpha2, data<half>(x2), alpha3, data<half>(x3),
							beta, data<half>(y), elements);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (elements % 4 == 0)
					kernel_axpy3<float, 4> <<<gridDim, blockDim, 0, stream>>>(alpha1, data<float>(x1), alpha2, data<float>(x2), alpha3,
							data<float>(x3), beta, data<float>(y), elements);
				else
					kernel_axpy3<float, 1> <<<gridDim, blockDim, 0, stream>>>(alpha1, data<float>(x1), alpha2, data<float>(x2), alpha3,
							data<float>(x3), beta, data<float>(y), elements);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_axpy3<double, 1> <<<gridDim, blockDim, 0, stream>>>(alpha1, data<double>(x1), alpha2, data<double>(x2), alpha3,
						data<double>(x3), beta, data<double>(y), elements);
				break;
			}
		}
		assert(cudaGetLastError() == cudaSuccess);
	}

	void cuda_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int length = volume(shape);
		dim3 blockDim(256);
		dim3 gridDim = ml::cuda_backend::gridSize<1024>(length, blockDim.x);
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

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
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

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
		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);

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

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
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

	void cuda_fused_act_bias_copy_backward(mlContext_t context, mlTensor_t dy, const mlTensor_t y, float beta_dx, mlTensor_t dx, float beta_db,
			mlTensor_t db, mlActivationType_t act)
	{
//		assert(same_shape(dy, dx));
//		assert(same_shape(dy, y));
		assert(dy.rank == 2 || dy.rank == 3);
//		if (dy.rank == 3)
//		{
//			assert(db.rank == 2);
//			assert(db.dim[0] == dy.dim[0]);
//			assert(db.dim[1] == dy.dim[2]);
//		}
//		else
//		{
//			assert(db.rank == 1);
//			assert(db.dim[0] == dy.dim[1]);
//		}

		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
		const int group_dim = (dy.rank == 3) ? dy.dim[0] : 1;
		const int first_dim = (dy.rank == 3) ? dy.dim[1] : dy.dim[0];
		const int last_dim = get_last_dim(dy);

//		assert(ml::cuda_backend::Context::getWorkspaceSize(context) >= last_dim * sizeof(float));
//		cudaStream_t stream = ml::cuda_backend::Context::getStream(context);
//		float *workspace = ml::cuda_backend::Context::getWorkspace<float>(context);
//		const size_t workspace_size = ml::cuda_backend::Context::getWorkspaceSize(context);
//		const int workspace_first_dim = std::min((size_t) 512, workspace_size / (sizeof(float) * group_dim * last_dim));

		const int hw_dim = std::min(1024, (first_dim + 7) / 8);

		dim3 blockDim1(32, 8);
		dim3 gridDim1_x1((last_dim + 31) / 32, hw_dim, group_dim);
		dim3 gridDim1_x4((last_dim + 127) / 127, hw_dim, group_dim);

//		float *dw_tmp_ptr = (db.data == nullptr) ? nullptr : workspace;

		cuda_scale_tensor(context, beta_db, db);

		switch (y.dtype)
		{
			case DTYPE_FLOAT16:
			{
				if (last_dim % 4 == 0)
					kernel_act_bias_copy_backward<half, 4, float> <<<gridDim1_x4, blockDim1, 0, stream>>>(data<half>(dy), data<half>(y), beta_dx,
							data<half>(dx), data<half>(db), first_dim, last_dim, act);
				else
					kernel_act_bias_copy_backward<half, 1, float> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<half>(dy), data<half>(y), beta_dx,
							data<half>(dx), data<half>(db), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim % 4 == 0)
					kernel_act_bias_copy_backward<float, 4> <<<gridDim1_x4, blockDim1, 0, stream>>>(data<float>(dy), data<float>(y), beta_dx,
							data<float>(dx), data<float>(db), first_dim, last_dim, act);
				else
					kernel_act_bias_copy_backward<float, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<float>(dy), data<float>(y), beta_dx,
							data<float>(dx), data<float>(db), first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT64:
			{
				kernel_act_bias_copy_backward<double, 1> <<<gridDim1_x1, blockDim1, 0, stream>>>(data<double>(dy), data<double>(y), beta_dx,
						data<double>(dx), data<double>(db), first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}
		assert(cudaGetLastError() == cudaSuccess);

//		dim3 blockDim2(32, 32);
//		dim3 gridDim2_x1((last_dim + 31) / 32);
//		dim3 gridDim2_x4((last_dim + 127) / 127);
//		if (db.data != nullptr)
//		{
//			switch (db.dtype)
//			{
//				case DTYPE_FLOAT16:
//				{
//					if (last_dim % 4 == 0)
//						kernel_sum_over_first_dim<float, 4, 2> <<<gridDim2_x4, blockDim2, 0, stream>>>(beta_db, data<half>(db), 1.0f, dw_tmp_ptr,
//								workspace_first_dim, last_dim);
//					else
//						kernel_sum_over_first_dim<float, 1, 2> <<<gridDim2_x1, blockDim2, 0, stream>>>(beta_db, data<half>(db), 1.0f, dw_tmp_ptr,
//								workspace_first_dim, last_dim);
//					break;
//				}
//				case DTYPE_FLOAT32:
//				{
//					if (last_dim % 4 == 0)
//						kernel_sum_over_first_dim<float, 4, 2> <<<gridDim2_x4, blockDim2, 0, stream>>>(beta_db, data<float>(db), 1.0f, dw_tmp_ptr,
//								workspace_first_dim, last_dim);
//					else
//						kernel_sum_over_first_dim<float, 1, 2> <<<gridDim2_x1, blockDim2, 0, stream>>>(beta_db, data<float>(db), 1.0f, dw_tmp_ptr,
//								workspace_first_dim, last_dim);
//					break;
//				}
//				default:
//					break;
//			}
//			assert(cudaGetLastError() == cudaSuccess);
//		}
	}
} /* namespace ml */
