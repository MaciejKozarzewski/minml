#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>
#include "utils.hpp"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstring>
#include <algorithm>
#include <inttypes.h>
#include <iostream>

namespace
{
#define tensorIdx4D(b,h,w,f) ((((b)*(height)+h)*(width)+w)*(filters)+f)
#define tensorIdx3D(b,p,f, plane, filters) (((b)*(plane)+p)*(filters)+f)
#define tensorIdx2D(h,w, width) ((h)*(width)+w)

	template<typename T>
	__global__ void kernel_conv3x3_4x4_weight_transform(T *matrices, const T *weights, const int filters, bool invert)
	{
		__shared__ T storage[18 * 32];

		T c23 = static_cast<T>(2.0 / 3.0);
		T c13 = static_cast<T>(1.0 / 3.0);
		T c2 = static_cast<T>(2);
		T c4 = static_cast<T>(4);
		for (int f = 0; f < filters; f += 32)
		{
			T load0 = 0, load1 = 0, load2 = 0;

			if (f + threadIdx.x < filters)
			{
				if (invert == false)
				{
					load0 = weights[(blockIdx.x * 9 + (threadIdx.y + 0 * 3)) * filters + f + threadIdx.x];
					load1 = weights[(blockIdx.x * 9 + (threadIdx.y + 1 * 3)) * filters + f + threadIdx.x];
					load2 = weights[(blockIdx.x * 9 + (threadIdx.y + 2 * 3)) * filters + f + threadIdx.x];
				}
				else
				{
					load0 = weights[(blockIdx.x * 9 + 8 - (threadIdx.y + 0 * 3)) * filters + f + threadIdx.x];
					load1 = weights[(blockIdx.x * 9 + 8 - (threadIdx.y + 1 * 3)) * filters + f + threadIdx.x];
					load2 = weights[(blockIdx.x * 9 + 8 - (threadIdx.y + 2 * 3)) * filters + f + threadIdx.x];
				}
			}

			int tmp_idx = threadIdx.y * 32 + threadIdx.x;
			storage[tmp_idx] = load0;
			storage[tmp_idx + 96] = c23 * (load0 + load1 + load2);
			storage[tmp_idx + 192] = c23 * (load0 - load1 + load2);
			storage[tmp_idx + 288] = c13 * (load0 + c2 * load1 + c4 * load2);
			storage[tmp_idx + 384] = c13 * (load0 - c2 * load1 + c4 * load2);
			storage[tmp_idx + 480] = c2 * load2;
			__syncthreads();

			for (int k = threadIdx.y; k < 6; k += 3)
			{
				tmp_idx = k * 96 + threadIdx.x;
				load0 = storage[tmp_idx];
				load1 = storage[tmp_idx + 32];
				load2 = storage[tmp_idx + 64];

				tmp_idx = ((6 * k + 0) * gridDim.x + blockIdx.x) * filters + f + threadIdx.x;
				if (f + threadIdx.x < filters)
				{
					matrices[tmp_idx + 0 * gridDim.x * filters] = load0;
					matrices[tmp_idx + 1 * gridDim.x * filters] = c23 * (load0 + load1 + load2);
					matrices[tmp_idx + 2 * gridDim.x * filters] = c23 * (load0 - load1 + load2);
					matrices[tmp_idx + 3 * gridDim.x * filters] = c13 * (load0 + c2 * load1 + c4 * load2);
					matrices[tmp_idx + 4 * gridDim.x * filters] = c13 * (load0 - c2 * load1 + c4 * load2);
					matrices[tmp_idx + 5 * gridDim.x * filters] = c2 * load2;
				}
			}
			__syncthreads();
		}
	}
	template<int tile_length, typename T>
	__launch_bounds__(384, 5)
	__global__ void kernel_conv3x3_4x4_input_transform(T *matrices, const T *input, int3 shape)
	{
		__shared__ float data[36][tile_length];

		for (int f = 0; f < shape.z; f += tile_length)
		{
			for (int i = threadIdx.y; i < 36; i += 6)
			{
				int h = 4 * blockIdx.y - 1 + i / 6;
				int w = 4 * blockIdx.z - 1 + i % 6;
				if (h >= 0 && h < shape.x && w >= 0 && w < shape.y)
				{
					int filter_id = f + threadIdx.x;
					int tmp_idx = ((blockIdx.x * shape.x + h) * shape.y + w) * shape.z + filter_id;
					if (filter_id < shape.z)
						data[i][threadIdx.x] = input[tmp_idx];
					if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
						data[i][threadIdx.x + blockDim.x] = input[tmp_idx + blockDim.x];
				}
				else
				{
					data[i][threadIdx.x] = 0.0f;
					if (threadIdx.x + blockDim.x < tile_length)
						data[i][threadIdx.x + blockDim.x] = 0.0f;
				}
			}
			__syncthreads();
			for (int i = threadIdx.x; i < tile_length; i += blockDim.x)
			{
				int tmp_idx = 6 * threadIdx.y;
				float load0 = data[tmp_idx + 0][i];
				float load1 = data[tmp_idx + 1][i];
				float load2 = data[tmp_idx + 2][i];
				float load3 = data[tmp_idx + 3][i];
				float load4 = data[tmp_idx + 4][i];
				float load5 = data[tmp_idx + 5][i];
				__syncthreads();

				data[tmp_idx + 0][i] = load0 - load2 + 0.25f * (load4 - load2);
				data[tmp_idx + 1][i] = load1 + load2 - 0.25f * (load3 + load4);
				data[tmp_idx + 2][i] = load2 - load1 + 0.25f * (load3 - load4);
				data[tmp_idx + 3][i] = load3 - load1 + 0.5f * (load4 - load2);
				data[tmp_idx + 4][i] = load1 - load3 + 0.5f * (load4 - load2);
				data[tmp_idx + 5][i] = load1 - load3 + 0.25f * (load5 - load3);
			}
			__syncthreads();
			for (int i = threadIdx.x; i < tile_length; i += blockDim.x)
			{
				int tmp_idx = threadIdx.y;
				float load0 = data[tmp_idx + 0][i];
				float load1 = data[tmp_idx + 6][i];
				float load2 = data[tmp_idx + 12][i];
				float load3 = data[tmp_idx + 18][i];
				float load4 = data[tmp_idx + 24][i];
				float load5 = data[tmp_idx + 30][i];
				__syncthreads();

				data[tmp_idx + 0][i] = load0 - load2 + 0.25f * (load4 - load2);
				data[tmp_idx + 6][i] = load1 + load2 - 0.25f * (load3 + load4);
				data[tmp_idx + 12][i] = load2 - load1 + 0.25f * (load3 - load4);
				data[tmp_idx + 18][i] = load3 - load1 + 0.5f * (load4 - load2);
				data[tmp_idx + 24][i] = load1 - load3 + 0.5f * (load4 - load2);
				data[tmp_idx + 30][i] = load1 - load3 + 0.25f * (load5 - load3);
			}
			__syncthreads();
			for (int i = threadIdx.y; i < 36; i += 6)
			{
				int filter_id = f + threadIdx.x;
				int tmp_idx = (((blockIdx.x + i * gridDim.x) * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.z + filter_id;
				if (filter_id < shape.z)
					matrices[tmp_idx] = data[i][threadIdx.x];
				if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
					matrices[tmp_idx + blockDim.x] = data[i][threadIdx.x + blockDim.x];
			}
			__syncthreads();
		}
	}
	template<int tile_length, typename T>
	__launch_bounds__(384, 5)
	__global__ void kernel_conv3x3_4x4_output_transform(const T *matrices, T *output, int3 shape, const T *biases, const T *add)
	{
		__shared__ float data[36][tile_length];

		for (int f = 0; f < shape.z; f += tile_length)
		{
			for (int i = threadIdx.y; i < 36; i += 6)
			{
				int filter_id = f + threadIdx.x;
				int tmp_idx = (((blockIdx.x + i * gridDim.x) * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.z + filter_id;
				if (filter_id < shape.z)
					data[i][threadIdx.x] = matrices[tmp_idx];
				if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
					data[i][threadIdx.x + blockDim.x] = matrices[tmp_idx + tile_length / 2];
			}
			__syncthreads();
			for (int i = 0; i < tile_length; i += blockDim.x)
			{
				int tmp_idx = 6 * threadIdx.y;
				float load0 = data[tmp_idx + 0][i + threadIdx.x];
				float load1 = data[tmp_idx + 1][i + threadIdx.x];
				float load2 = data[tmp_idx + 2][i + threadIdx.x];
				float load3 = data[tmp_idx + 3][i + threadIdx.x];
				float load4 = data[tmp_idx + 4][i + threadIdx.x];
				float load5 = data[tmp_idx + 5][i + threadIdx.x];
				__syncthreads();

				data[tmp_idx + 0][i + threadIdx.x] = load0 + load1 + load2 + 0.25f * (load3 + load4);
				data[tmp_idx + 1][i + threadIdx.x] = load1 - load2 + 0.5f * (load3 - load4);
				data[tmp_idx + 2][i + threadIdx.x] = load1 + load2 + load3 + load4;
				data[tmp_idx + 3][i + threadIdx.x] = load1 - load2 + 2.0f * (load3 - load4 + load5);
			}
			__syncthreads();
			for (int i = 0; i < tile_length; i += blockDim.x)
			{
				float bias = 0.0f;
				if (biases != nullptr && (f + i + threadIdx.x) < shape.z)
					bias = biases[f + i + threadIdx.x];

				float load0, load1, load2, load3, load4, load5;
				if (threadIdx.y < 4)
				{
					load0 = data[threadIdx.y + 0][i + threadIdx.x];
					load1 = data[threadIdx.y + 6][i + threadIdx.x];
					load2 = data[threadIdx.y + 12][i + threadIdx.x];
					load3 = data[threadIdx.y + 18][i + threadIdx.x];
					load4 = data[threadIdx.y + 24][i + threadIdx.x];
					load5 = data[threadIdx.y + 30][i + threadIdx.x];
				}
				__syncthreads();
				if (threadIdx.y < 4)
				{
					data[threadIdx.y + 0][i + threadIdx.x] = bias + load0 + load1 + load2 + 0.25f * (load3 + load4);
					data[threadIdx.y + 4][i + threadIdx.x] = bias + load1 - load2 + 0.5f * (load3 - load4);
					data[threadIdx.y + 8][i + threadIdx.x] = bias + load1 + load2 + load3 + load4;
					data[threadIdx.y + 12][i + threadIdx.x] = bias + load1 - load2 + 2.0f * (load3 - load4 + load5);
				}
			}
			__syncthreads();
			if (threadIdx.y < 4)
				for (int i = threadIdx.y; i < 16; i += 4)
				{
					int h = 4 * blockIdx.y + i / 4;
					int w = 4 * blockIdx.z + i % 4;
					if (h < shape.x && w < shape.y)
					{
						int filter_id = f + threadIdx.x;
						int tmp_idx = ((blockIdx.x * shape.x + h) * shape.y + w) * shape.z + filter_id;

						if (filter_id < shape.z)
						{
							if (add != nullptr)
								data[i][threadIdx.x] += static_cast<float>(add[tmp_idx]);
							output[tmp_idx] = data[i][threadIdx.x];
						}
						if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
						{
							if (add != nullptr)
								data[i][threadIdx.x + blockDim.x] += static_cast<float>(add[tmp_idx + blockDim.x]);
							output[tmp_idx + blockDim.x] = data[i][threadIdx.x + blockDim.x];
						}
					}
				}
			__syncthreads();
		}
	}
	__global__ void kernel_conv3x3_4x4_gradient_transform(float *matrices, const float *gradient, int height, int width, int filters)
	{
		__shared__ int indices_in[16];
		__shared__ int indices_out[36];
		__shared__ float data[36 * 32];
		int tid = 32 * threadIdx.y + threadIdx.x; //192 threads

		if (tid < 16)
		{
			int h = 4 * blockIdx.y + tid / 4;
			int w = 4 * blockIdx.z + tid % 4;
			if (h < height && w < width)
				indices_in[tid] = tensorIdx4D(blockIdx.x, h, w, 0);
			else
				indices_in[tid] = -1;
		}
		if (tid < 36)
		{
			indices_out[tid] = (((blockIdx.x + tid * gridDim.x) * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * filters;
		}
		for (int f = threadIdx.x; f < filters; f += 32)
		{
			__syncthreads();
			for (int i = threadIdx.y; i < 16; i += 6)
				if (indices_in[i] != -1)
					data[32 * i + threadIdx.x] = gradient[indices_in[i] + f];
				else
					data[32 * i + threadIdx.x] = 0.0f;
			__syncthreads();
			float load0, load1, load2, load3;
			if (threadIdx.y < 4)
			{
				int tmp_idx = 32 * threadIdx.y + threadIdx.x;
				load0 = data[tmp_idx + 0];
				load1 = data[tmp_idx + 128];
				load2 = data[tmp_idx + 256];
				load3 = data[tmp_idx + 384];
			}
			__syncthreads();

			if (threadIdx.y < 4)
			{
				int tmp_idx = 32 * threadIdx.y + threadIdx.x;
				float tmp0 = 2.0f * load3; //2*load3
				float tmp1 = load0 + load2; //load0+load2
				float tmp2 = load1 + load3; //load1+load3
				float tmp3 = 0.333333f * tmp1;
				tmp1 = 0.666667f * tmp1;
				tmp2 = 0.666667f * tmp2;
				tmp3 = tmp3 + load2;
				float tmp4 = tmp2 + tmp0;

				data[tmp_idx] = load0;
				data[tmp_idx + 128] = tmp1 + tmp2;
				data[tmp_idx + 256] = tmp1 - tmp2;
				data[tmp_idx + 384] = tmp3 + tmp4;
				data[tmp_idx + 512] = tmp3 - tmp4;
				data[tmp_idx + 640] = tmp0;
			}
			__syncthreads();

			int tmp_idx = 128 * threadIdx.y + threadIdx.x;
			load0 = data[tmp_idx];
			load1 = data[tmp_idx + 32];
			load2 = data[tmp_idx + 64];
			load3 = data[tmp_idx + 96];

			float tmp0 = 2.0f * load3; //2*load3
			float tmp1 = load0 + load2; //load0+load2
			float tmp2 = load1 + load3; //load1+load3
			float tmp3 = 0.333333f * tmp1;
			tmp1 = 0.666667f * tmp1;
			tmp2 = 0.666667f * tmp2;
			tmp3 = tmp3 + load2;
			float tmp4 = tmp2 + tmp0;

			matrices[indices_out[6 * threadIdx.y + 0] + f] = load0;
			matrices[indices_out[6 * threadIdx.y + 1] + f] = tmp1 + tmp2;
			matrices[indices_out[6 * threadIdx.y + 2] + f] = tmp1 - tmp2;
			matrices[indices_out[6 * threadIdx.y + 3] + f] = tmp3 + tmp4;
			matrices[indices_out[6 * threadIdx.y + 4] + f] = tmp3 - tmp4;
			matrices[indices_out[6 * threadIdx.y + 5] + f] = tmp0;
		}
	}
	__global__ void kernel_conv3x3_4x4_update_transform(const float *matrices, float *update, int filters)
	{
		__shared__ int indices_in[36];
		__shared__ float data[36 * 32];
		int tid = 32 * threadIdx.y + threadIdx.x; //192 threads
		if (tid < 36)
			indices_in[tid] = (tid * gridDim.x + blockIdx.x) * filters;

		for (int f = threadIdx.x; f < filters; f += 32)
		{
			__syncthreads();
			for (int i = threadIdx.y; i < 36; i += 6)
				data[32 * i + threadIdx.x] = matrices[indices_in[i] + f];
			__syncthreads();

			int tmp_idx = 32 * threadIdx.y + threadIdx.x;
			float load0 = data[tmp_idx];
			float load1 = data[tmp_idx + 192];
			float load2 = data[tmp_idx + 384];
			float load3 = data[tmp_idx + 576];
			float load4 = data[tmp_idx + 768];
			float load5 = data[tmp_idx + 960];

			float tmp1 = load1 + load2;
			float tmp2 = load1 - load2;
			float tmp3 = load3 + load4;
			float tmp4 = load3 - load4;
			load0 += tmp1 + 0.25f * tmp3;
			load1 = tmp2 + 0.5f * tmp4;
			load2 = tmp1 + tmp3 + 2.0f * load5;

			__syncthreads();
			data[tmp_idx] = load0;
			data[tmp_idx + 192] = load1;
			data[tmp_idx + 384] = load2;
			__syncthreads();

			if (threadIdx.y < 3)
			{
				tmp_idx = 192 * threadIdx.y + threadIdx.x;
				load0 = data[tmp_idx];
				load1 = data[tmp_idx + 32];
				load2 = data[tmp_idx + 64];
				load3 = data[tmp_idx + 96];
				load4 = data[tmp_idx + 128];
				load5 = data[tmp_idx + 160];

				tmp1 = load1 + load2;
				tmp2 = load1 - load2;
				tmp3 = load3 + load4;
				tmp4 = load3 - load4;

				load0 += tmp1 + 0.25f * tmp3;
				load1 = tmp2 + 0.5f * tmp4;
				load2 = tmp1 + tmp3 + 2.0f * load5;

				update[(blockIdx.x * 9 + 3 * threadIdx.y + 0) * filters + f] += load0;
				update[(blockIdx.x * 9 + 3 * threadIdx.y + 1) * filters + f] += load1;
				update[(blockIdx.x * 9 + 3 * threadIdx.y + 2) * filters + f] += load2;
			}
		}
	}

}
namespace ml
{
//		int cuda_winograd3x3_4x4_transform_weight(cudaStream_t stream, ConstTensorDescriptor *weight, TensorDescriptor *matrices, bool invert)
//		{
//			int filters_out = weight->shape[0];
//			int filters_in = weight->shape[3];
//			dim3 blockSize(32, 3);
//			dim3 gridSize(filters_out);
//
//			switch (weight->dtype)
//			{
////				case DTYPE_FLOAT16:
////					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
////							reinterpret_cast<const half*>(weight->data), filters_in, invert);
////					break;
//				case DTYPE_FLOAT32:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<float*>(matrices->data),
//							reinterpret_cast<const float*>(weight->data), filters_in, invert);
//					break;
//				case DTYPE_FLOAT64:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<double*>(matrices->data),
//							reinterpret_cast<const double*>(weight->data), filters_in, invert);
//					break;
//			}
//			return cudaGetLastError();
//		}
	int cuda_winograd3x3_4x4_transform_input(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
		int3 shape { input_shape.dim[1], input_shape.dim[2], input_shape.dim[3] };

		int tiles_h = (input_shape.dim[1] + 3) / 4;
		int tiles_w = (input_shape.dim[2] + 3) / 4;
		dim3 gridSize(input_shape.dim[0], tiles_h, tiles_w);
		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockSize(32, 6);
		kernel_conv3x3_4x4_input_transform<64> <<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<float*>(matrices),
				reinterpret_cast<const float*>(input), shape);
		return cudaGetLastError();
	}
	int cuda_winograd3x3_4x4_transform_output(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		int3 shape { output_shape.dim[1], output_shape.dim[2], output_shape.dim[3] };

		int tiles_h = (output_shape.dim[1] + 3) / 4;
		int tiles_w = (output_shape.dim[2] + 3) / 4;
		dim3 gridSize(output_shape.dim[0], tiles_h, tiles_w);
		cudaStream_t stream = cuda::Context::getStream(context);

		dim3 blockSize(32, 6);
		kernel_conv3x3_4x4_output_transform<64> <<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<const float*>(matrices),
				reinterpret_cast<float*>(output), shape, reinterpret_cast<const float*>(bias), reinterpret_cast<const float*>(add));
		return cudaGetLastError();
	}
//		int cuda_winograd3x3_4x4_transform_gradient(cudaStream_t stream, ConstTensorDescriptor *gradient_next, TensorDescriptor *matrices)
//		{
//			int tiles_h = (gradient_next->shape[1] + 3) / 4;
//			int tiles_w = (gradient_next->shape[2] + 3) / 4;
//			dim3 blockSize(32, 6);
//			dim3 gridSize(gradient_next->shape[0], tiles_h, tiles_w);
//
//			kernel_conv3x3_4x4_gradient_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<float*>(matrices->data),
//					reinterpret_cast<const float*>(gradient_next->data), gradient_next->shape[1], gradient_next->shape[2], gradient_next->shape[3]);
//			return cudaGetLastError();
//		}
//		int cuda_winograd3x3_4x4_transform_update(cudaStream_t stream, TensorDescriptor *update, ConstTensorDescriptor *matrices)
//		{
//			dim3 blockSize(32, 6);
//			dim3 gridSize(update->shape[0]);
//			kernel_conv3x3_4x4_update_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<const float*>(matrices->data),
//					reinterpret_cast<float*>(update->data), update->shape[3]);
//			return cudaGetLastError();
//		}

} /* namespace ml */

