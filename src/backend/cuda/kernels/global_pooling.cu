/*
 * global_pooling.cu
 *
 *  Created on: Feb 16, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "../helpers/tensor_wrappers.cuh"
#include "../vectors/vectors.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace
{
	using namespace vectors;
	using namespace ml;

//	__global__ void kernel_pooling_forward_avg_max(const float *__restrict__ input, int dim0, int dim1, int dim2, int *max_indices)
//	{
//		assert(blockDim.x == 32 && blockDim.y == 32);
//		__shared__ int shared_indices[32 * 32];
//		__shared__ float shared_max[32 * 32];
//
//		const int last_dim_index = blockIdx.x * 32 + threadIdx.x;
//		const int idx = threadIdx.y * 32 + threadIdx.x;
//
//		Indexer<3> input_indexer(dim0, dim1, dim2);
//
//		float local_max(-1000.0f);
//		int local_max_index = 0;
//		for (int i = threadIdx.y; i < dim1; i += 32)
//			if (last_dim_index < dim2)
//			{
//				const float tmp = input[input_indexer.at(blockIdx.z, i, last_dim_index)];
//				if (tmp > local_max)
//				{
//					local_max = tmp;
//					local_max_index = i;
//				}
//			}
//
//		shared_max[idx] = local_max;
//		__syncthreads();
//
//		for (int i = 16; i >= 1; i /= 2)
//		{
//			if (threadIdx.y < i)
//			{
//				shared_avg[idx] += shared_avg[idx + i * 32];
//				shared_max[idx] = max(shared_max[idx + i * 32], shared_max[idx + i * 32]);
//			}
//			__syncthreads();
//		}
//
//		TensorWrapper<2, T> output_wrapper(output, dim0, 2 * dim2);
//		if (threadIdx.y == 0)
//		{
//			const Vector<T> inv(1.0f / static_cast<float>(dim1));
//			local_avg = shared_avg[threadIdx.x] * inv;
//			local_max = shared_max[threadIdx.x];
//
//			output_wrapper.store(local_avg, blockIdx.z, last_dim_index);
//			output_wrapper.store(local_max, blockIdx.z, dim2 + last_dim_index);
//		}
//	}
	template<typename T>
	__global__ void kernel_pooling_forward_avg_max(T *__restrict__ output, const T *__restrict__ input, int dim0, int dim1, int dim2)
	{
		assert(blockDim.x == 32 && blockDim.y == 32);
		__shared__ Vector<T> shared_avg[32 * 32];
		__shared__ Vector<T> shared_max[32 * 32];

		const int last_dim_index = blockIdx.x * 32 * vector_length<T>() + threadIdx.x;
		const int idx = threadIdx.y * 32 + threadIdx.x;

		ConstTensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);

		Vector<T> local_avg = vector_zero<T>();
		Vector<T> local_max(-1000.0f);
		int max_index = 0;
		for (int i = threadIdx.y; i < dim1; i += 32)
		{
			const Vector<T> tmp = input_wrapper.load(blockIdx.z, i, last_dim_index);
			local_avg += tmp;
			local_max = max(local_max, tmp);
		}

		shared_avg[idx] = local_avg;
		shared_max[idx] = local_max;
		__syncthreads();

		for (int i = 16; i >= 1; i /= 2)
		{
			if (threadIdx.y < i)
			{
				shared_avg[idx] += shared_avg[idx + i * 32];
				shared_max[idx] = max(shared_max[idx + i * 32], shared_max[idx + i * 32]);
			}
			__syncthreads();
		}

		TensorWrapper<3, T> output_wrapper(output, dim0, 2, dim2);
		if (threadIdx.y == 0)
		{
			const Vector<T> inv(1.0f / static_cast<float>(dim1));
			local_avg = shared_avg[threadIdx.x] * inv;
			local_max = shared_max[threadIdx.x];

			output_wrapper.store(local_avg, blockIdx.z, 0, last_dim_index);
			output_wrapper.store(local_max, blockIdx.z, 1, last_dim_index);
		}
	}
	template<typename T>
	__global__ void kernel_broadcast(T *input, const T *bias, int dim0, int dim1, int dim2)
	{
		ConstTensorWrapper<1, T> bias_wrapper(bias, dim2);

		TensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);
		for (int j = (blockIdx.x * blockDim.x + threadIdx.x) * vector_length<T>(); j < dim2; j += gridDim.x * blockDim.x * vector_length<T>())
		{
			const Vector<T> _bias = bias_wrapper.load(j);
			for (int i = blockIdx.y; i < dim1; i += gridDim.y)
			{
				const Vector<T> tmp = input_wrapper.load(blockIdx.z, i, j) + _bias;
				input_wrapper.store(tmp, blockIdx.z, i, j);
			}
		}
	}
}

namespace ml
{
	void cuda_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output,
			const void *weights)
	{
		const int dim0 = shape.dim[0];
		const int dim1 = shape.dim[1] * shape.dim[2];
		const int dim2 = shape.dim[3];

		dim3 blockDim(32, 32);
		dim3 gridDim((dim2 + 31) / 32, 1, dim0);

		cudaStream_t stream = cuda::Context::getStream(context);

		const int workspace_size = dim0 * 2 * dim2 * size_of(dtype);
		assert(cuda::Context::getWorkspaceSize(context) >= 2 *workspace_size);
		void *pooled_1 = cuda::Context::getWorkspace<uint8_t>(context);
		void *pooled_2 = cuda::Context::getWorkspace<uint8_t>(context) + workspace_size;

		switch (dtype)
		{
//			case DTYPE_BFLOAT16:
//				kernel_pooling_forward_avg_max<<<gridDim, blockDim, 0, stream >>>(reinterpret_cast<__nv_bfloat16* >(pooled_1),
//						getPointer<__nv_bfloat16 >(input), dim0, dim1, dim2);
//				break;
			case DTYPE_FLOAT16:
				kernel_pooling_forward_avg_max<<<gridDim, blockDim, 0, stream >>>(reinterpret_cast<half*>(pooled_1), getPointer<half>(input), dim0,
						dim1, dim2);
				break;
			case DTYPE_FLOAT32:
				kernel_pooling_forward_avg_max<<<gridDim, blockDim, 0, stream >>>(reinterpret_cast<float*>(pooled_1), getPointer<float>(input), dim0,
						dim1, dim2);
				break;
		}
		mlShape_t in_shape = make_shape( { dim0, dim2 });
		mlShape_t w_shape = make_shape( { dim2, dim2 });
		mlShape_t out_shape = make_shape( { dim0, dim2 });

		cuda_gemm(context, dtype, out_shape, pooled_2, in_shape, pooled_1, w_shape, weights, 'n', 't', 1.0f, 0.0f);

		dim3 gridDim2((dim2 + 127) / 128, 1, dim0);
		switch (dtype)
		{
//			case DTYPE_BFLOAT16:
//				kernel_broadcast<<<gridDim2, 128, 0, stream >>>(getPointer<__nv_bfloat16 >(output), reinterpret_cast<__nv_bfloat16* >(pooled_2),
//						dim0, dim1, dim2);
//				break;
			case DTYPE_FLOAT16:
				kernel_broadcast<<<gridDim2, 128, 0, stream >>>(getPointer<half>(output), reinterpret_cast<half*>(pooled_2), dim0, dim1, dim2);
				break;
			case DTYPE_FLOAT32:
				kernel_broadcast<<<gridDim2, 128, 0, stream >>>(getPointer<float>(output), reinterpret_cast<float*>(pooled_2), dim0, dim1, dim2);
				break;
		}

		assert(cudaGetLastError() == cudaSuccess);
	}
	void cuda_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *input, const void *weights)
	{
	}

} /* namespace ml */

