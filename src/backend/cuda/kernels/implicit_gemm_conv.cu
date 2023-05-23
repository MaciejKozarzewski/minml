/*
 * implicit_gemm_conv.cu
 *
 *  Created on: Jan 7, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.cuh"
#include "../utils.hpp"

#include "../helpers/accumulators.cuh"
#include "../helpers/indexers.cuh"
#include "../helpers/tensor_wrappers.cuh"
#include "../helpers/lines_and_tiles.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <iostream>

namespace
{
	using namespace ml;
	using namespace vectors;

//	__device__ constexpr int square(int x)
//	{
//		return x * x;
//	}
//	__device__ int2 get_number_of_tiles(int height, int width, int tile_size)
//	{
//		const int h = (height + tile_size - 1) / tile_size;
//		const int w = (width + tile_size - 1) / tile_size;
//		return int2 { h, w };
//	}
//
////	template<int GroupSize>
////	__device__ int2 group_index()
////	{
////		return int2 { threadIdx.x / GroupSize, threadIdx.x % GroupSize };
////	}
//
//	__device__ int grid_volume()
//	{
//		return gridDim.x * gridDim.y * gridDim.z;
//	}
//
//	template<int X>
//	__device__ dim3 reshape_thread_block(int thread_idx)
//	{
//		assert(blockDim.x % X == 0);
//		dim3 result;
//		result.x = thread_idx % X;
//		result.y = thread_idx / X;
//		result.z = 0;
//		return result;
//	}
//	template<int X, int Y, int Z = 1>
//	__device__ dim3 reshape_thread_block(int thread_idx)
//	{
//		assert(X * Y * Z == blockDim.x);
//		dim3 result;
//		result.x = thread_idx % X;
//		result.y = (thread_idx / X) % Y;
//		result.z = thread_idx / (X * Y);
//		return result;
//	}
//
//	class PaddedIndexer
//	{
//		public:
//			int stride0, stride1;
//#ifndef NDEBUG
//			int d0, d1, d2;
//#endif
//		public:
//			__device__ PaddedIndexer() // @suppress("Class members should be properly initialized")
//			{
//			}
//			__device__ PaddedIndexer(int dim0, int dim1, int dim2) :
//					stride0(dim1 * dim2),
//					stride1(dim2)
//			{
//#ifndef NDEBUG
//				d0 = dim0;
//				d1 = dim1;
//				d2 = dim2;
//#endif
//			}
//			__device__ int last_dim() const
//			{
//				return stride1;
//			}
//			__device__ int at(int x0, int x1, int x2) const
//			{
//				assert(0 <= x0 && x0 < d0);
//				assert(0 <= x1 && x1 < d1);
//				assert(0 <= x2 && x2 < d2);
//				const int result = x0 * stride0 + x1 * stride1 + x2;
//				return result; // + 8 * (result / 32); // add padding of one elements per 32
//			}
//			__device__ constexpr int rank() const
//			{
//				return 3;
//			}
//	};
//
//	template<int N>
//	class PaddedIndexer_v3
//	{
//		public:
//			int stride0;
//#ifndef NDEBUG
//			int d0, d1;
//#endif
//		public:
//			__device__ PaddedIndexer_v3() // @suppress("Class members should be properly initialized")
//			{
//			}
//			__device__ PaddedIndexer_v3(int dim0, int dim1) :
//					stride0(dim1)
//			{
//#ifndef NDEBUG
//				d0 = dim0;
//				d1 = dim1;
//#endif
//			}
//			__device__ int last_dim() const
//			{
//				return stride0;
//			}
//			__device__ int at(int x0, int x1) const
//			{
//				assert(0 <= x0 && x0 < d0);
//				assert(0 <= x1 && x1 < d1);
//				const int result = x0 * stride0 + x1;
//				return result + 2 * (result / N); // add padding every N elements
//			}
//			__device__ constexpr int rank() const
//			{
//				return 2;
//			}
//	};
//
//	__device__ float4 zero_x4()
//	{
//		float4 result;
//		result.x = 0;
//		result.y = 0;
//		result.z = 0;
//		result.w = 0;
//		return result;
//	}
//	__device__ float4 one_x4()
//	{
//		float4 result;
//		result.x = 1;
//		result.y = 1;
//		result.z = 1;
//		result.w = 1;
//		return result;
//	}
//
//	struct Accumulator
//	{
//			float4 x0, x1, x2, x3;
//
//			__device__ Accumulator()
//			{
//				x0 = zero_x4();
//				x1 = zero_x4();
//				x2 = zero_x4();
//				x3 = zero_x4();
//			}
//			__device__ void fma(const float4 &lhs, const float4 &rhs)
//			{
//				x0.x += lhs.x * rhs.x;
//				x0.y += lhs.x * rhs.y;
//				x0.z += lhs.x * rhs.z;
//				x0.w += lhs.x * rhs.w;
//
//				x1.x += lhs.y * rhs.x;
//				x1.y += lhs.y * rhs.y;
//				x1.z += lhs.y * rhs.z;
//				x1.w += lhs.y * rhs.w;
//
//				x2.x += lhs.z * rhs.x;
//				x2.y += lhs.z * rhs.y;
//				x2.z += lhs.z * rhs.z;
//				x2.w += lhs.z * rhs.w;
//
//				x3.x += lhs.w * rhs.x;
//				x3.y += lhs.w * rhs.y;
//				x3.z += lhs.w * rhs.z;
//				x3.w += lhs.w * rhs.w;
//			}
//			__device__ float4 operator [](int row) const
//			{
//				assert(0 <= row && row < 4);
//				return reinterpret_cast<const float4*>(this)[row];
//			}
//			__device__ void add(const Accumulator &other)
//			{
//				x0.x += other.x0.x;
//				x0.y += other.x0.y;
//				x0.z += other.x0.z;
//				x0.w += other.x0.w;
//
//				x1.x += other.x1.x;
//				x1.y += other.x1.y;
//				x1.z += other.x1.z;
//				x1.w += other.x1.w;
//
//				x2.x += other.x2.x;
//				x2.y += other.x2.y;
//				x2.z += other.x2.z;
//				x2.w += other.x2.w;
//
//				x3.x += other.x3.x;
//				x3.y += other.x3.y;
//				x3.z += other.x3.z;
//				x3.w += other.x3.w;
//			}
//	};
//
//	template<typename T, int N, int M>
//	struct LineLoader
//	{
//			__device__ Line<T, N> load(const T *src, int stride) const
//			{
//				return Line<T, N>();
//			}
//	};
//
//	template<typename T>
//	struct LineLoader<T, 8, 2>
//	{
//			__device__ Line<T, 8> load(const T *src, int stride) const
//			{
//				Line<T, 8> result;
//				reinterpret_cast<int2*>(&result)[0] = *reinterpret_cast<const int2*>(src + 0 * stride);
//				reinterpret_cast<int2*>(&result)[1] = *reinterpret_cast<const int2*>(src + 1 * stride);
//				reinterpret_cast<int2*>(&result)[2] = *reinterpret_cast<const int2*>(src + 2 * stride);
//				reinterpret_cast<int2*>(&result)[3] = *reinterpret_cast<const int2*>(src + 3 * stride);
//				return result;
//			}
//			__device__ void store(const Line<T, 8> &line, T *src, int stride) const
//			{
//				*reinterpret_cast<int2*>(src + 0 * stride) = reinterpret_cast<const int2*>(&line)[0];
//				*reinterpret_cast<int2*>(src + 1 * stride) = reinterpret_cast<const int2*>(&line)[1];
//				*reinterpret_cast<int2*>(src + 2 * stride) = reinterpret_cast<const int2*>(&line)[2];
//				*reinterpret_cast<int2*>(src + 3 * stride) = reinterpret_cast<const int2*>(&line)[3];
//			}
//	};
//	template<typename T>
//	struct LineLoader<T, 8, 4>
//	{
//			__device__ Line<T, 8> load(const T *src, int stride) const
//			{
//				Line<T, 8> result;
//				reinterpret_cast<int4*>(&result)[0] = reinterpret_cast<const int4*>(src + 0 * stride);
//				reinterpret_cast<int4*>(&result)[1] = reinterpret_cast<const int4*>(src + 1 * stride);
//				return result;
//			}
//			__device__ void store(const Line<T, 8> &line, T *src, int stride) const
//			{
//				*reinterpret_cast<int4*>(src + 0 * stride) = reinterpret_cast<const int4*>(&line)[0];
//				*reinterpret_cast<int4*>(src + 1 * stride) = reinterpret_cast<const int4*>(&line)[1];
//			}
//	};
//
//	template<int KernelSize, int InputTile, int OutputTile, int FilterTile, typename T>
//	__launch_bounds__(256, 2)
//	__global__ void kernel_implicit_gemm_v4(const T *input, const T *weights, T *output, int batch_size, int height, int width, int input_filters,
//			int output_filters)
//	{
//		static_assert(FilterTile >= 16, "");
//		static_assert(InputTile % 32 == 0, "");
//		static_assert(OutputTile % 32 == 0, "");
//		assert(blockDim.x == 256);
//		constexpr int Padding = KernelSize / 2;
//
//		__shared__ T input_workspace[(InputTile + 2) * FilterTile * vector_length<T>()];
//		__shared__ T weights_workspace[(OutputTile + 2) * FilterTile * vector_length<T>()];
//		__shared__ int spatial_offsets[3][InputTile];
//
//		for (int i = threadIdx.x; i < InputTile; i += blockDim.x)
//		{
//			const int main_index = blockIdx.y * InputTile + i;
//			spatial_offsets[0][i] = main_index / (height * width);
//			spatial_offsets[1][i] = (main_index / width) % height;
//			spatial_offsets[2][i] = main_index % width;
//		}
//		__syncthreads();
//
//		Tile<T, InputTile / 16, OutputTile / 16> accumulator;
//		accumulator.fill(0.0f);
//
//		for (int filter_in = 0; filter_in < input_filters; filter_in += FilterTile)
//		{
//			for (int kernel_row = 0; kernel_row < KernelSize; kernel_row++)
//				for (int kernel_col = 0; kernel_col < KernelSize; kernel_col++)
//				{
//					cg::thread_block_tile<FilterTile, cg::thread_block> sub_block = cg::tiled_partition<FilterTile>(cg::this_thread_block());
//
//					// loading input
//					PaddedIndexer_v3<InputTile> i_workspace_indexer(FilterTile, InputTile);
//					ConstTensorWrapper<4, T> input_wrapper(input, batch_size, height, width, input_filters);
//					for (int i = sub_block.meta_group_rank(); i < InputTile; i += sub_block.meta_group_size())
//						for (int j = sub_block.thread_rank(); j < FilterTile; j += sub_block.size())
//						{
//							const int b = spatial_offsets[0][i];
//							const int h = spatial_offsets[1][i] - Padding + kernel_row;
//							const int w = spatial_offsets[2][i] - Padding + kernel_col;
//							const int f = filter_in + j;
//
//							T tmp = 0.0f; //vector_zero<T>();
//							if (0 <= h and h < height and 0 <= w and w < width and b < batch_size and f < input_filters)
//								tmp = input_wrapper.load(b, h, w, f);
//							input_workspace[i_workspace_indexer.at(j, i)] = tmp;
//						}
//
//					// loading weights
//					ConstTensorWrapper<4, T> weights_wrapper(weights, output_filters, KernelSize, KernelSize, input_filters);
//					PaddedIndexer_v3<OutputTile> w_workspace_indexer(FilterTile, OutputTile);
//					for (int i = sub_block.meta_group_rank(); i < OutputTile; i += sub_block.meta_group_size())
//						for (int j = sub_block.thread_rank(); j < FilterTile; j += sub_block.size())
//						{
//							const int f_out = blockIdx.x * OutputTile + i;
//							const int f_in = filter_in + j;
//
//							T tmp = 0.0f;
//							if (f_out < output_filters and f_in < input_filters)
//								tmp = weights_wrapper.load(f_out, kernel_row, kernel_col, f_in);
//							weights_workspace[w_workspace_indexer.at(j, i)] = tmp;
//						}
//					__syncthreads();
//
//					// calculate outer products
//					const T *ptr_input = input_workspace + 2 * (threadIdx.x / 16);
//					const T *ptr_weights = weights_workspace + 2 * (threadIdx.x % 16);
//#pragma unroll 4
//					for (int k = 0; k < FilterTile; k++)
//					{
//						const Line<T, accumulator.rows()> lhs = LineLoader<T, accumulator.rows(), 2>().load(ptr_input, 32);
//						const Line<T, accumulator.columns()> rhs = LineLoader<T, accumulator.columns(), 2>().load(ptr_weights, 32);
////						Line<T, accumulator.rows()> lhs;
////						Line<T, accumulator.columns()> rhs;
////						lhs.fill(1.0f);
////						rhs.fill(1.0f);
//
//						outer_product(accumulator, lhs, rhs);
//						ptr_input += InputTile + 2;
//						ptr_weights += OutputTile + 2;
//					}
//					__syncthreads();
//				}
//		}
//
//		cg::thread_block_tile<16, cg::thread_block> sub_block = cg::tiled_partition<16>(cg::this_thread_block());
//		PaddedIndexer_v3<OutputTile> workspace_indexer(16, OutputTile);
//		Indexer<4> output_indexer(batch_size, height, width, output_filters);
//		for (int row = 0; row < accumulator.rows(); row++)
//		{
//			T *ptr = weights_workspace + workspace_indexer.at(sub_block.meta_group_rank(), 2 * sub_block.thread_rank());
//			LineLoader<T, accumulator.rows(), 2>().store(accumulator.get_row(row), ptr, 32);
//			sub_block.sync();
//
//			for (int j = sub_block.thread_rank(); j < OutputTile; j += sub_block.num_threads())
//			{
//				const int out_s = (row / 2) * 32 + row % 2 + 2 * sub_block.meta_group_rank();
//				const int b = spatial_offsets[0][out_s];
//				const int h = spatial_offsets[1][out_s];
//				const int w = spatial_offsets[2][out_s];
//				const int f_out = blockIdx.x * OutputTile + j;
//				if (h < height and w < width and b < batch_size and f_out < output_filters)
//					output[output_indexer.at(b, h, w, f_out)] = ptr[j];
//			}
//		}
//	}
//
//	template<int KernelSize, int InputTile, int OutputTile, int FilterTile, typename T>
//	__launch_bounds__(256, 2)
//	__global__ void kernel_implicit_gemm_v3(const T *input, const T *weights, T *output, int batch_size, int height, int width, int input_filters,
//			int output_filters)
//	{
//		assert(blockDim.x == 256);
//		constexpr int Padding = KernelSize / 2;
//
//		__shared__ T workspace[(InputTile + OutputTile + 4) * FilterTile * vector_length<T>()];
//		T *input_workspace = workspace; // [InputTile * FilterTile * vector_length<T>()];
//		T *weights_workspace = workspace + (InputTile + 2) * FilterTile * vector_length<T>(); // [OutputTile * FilterTile * vector_length<T>()];
//		__shared__ int spatial_offsets[InputTile];
//
//		for (int i = threadIdx.x; i < InputTile; i += blockDim.x)
//		{
//			const Indexer<4> indexer(batch_size, height, width, input_filters);
//			const int main_index = blockIdx.y * InputTile + i;
//			const int b = main_index / (height * width);
//			const int h = (main_index / width) % height;
//			const int w = main_index % width;
//			if (0 <= h and h < height and 0 <= w and w < width and b < batch_size)
//				spatial_offsets[i] = indexer.at(b, h, w, 0);
//			else
//				spatial_offsets[i] = -1;
//		}
//		__syncthreads();
//
////		const Indexer<4> indexer(batch_size, height, width, input_filters);
////		const int main_index = blockIdx.y * InputTile + i;
////		const int b = main_index / (height * width);
////		const int h = (main_index / width) % height - Padding + kernel_row;
////		const int w = main_index % width - Padding + kernel_col;
////		if (0 <= h and h < height and 0 <= w and w < width and b < batch_size)
////			spatial_offsets[i] = indexer.at(b, h, w, 0);
////		else
////			spatial_offsets[i] = -1;
//
//		Tile<T, 8, 8> accumulator;
//		accumulator.fill(0.0f);
//		for (int kernel_row = 0; kernel_row < KernelSize; kernel_row++)
//			for (int kernel_col = 0; kernel_col < KernelSize; kernel_col++)
//			{
//
//				for (int filter_in = 0; filter_in < input_filters; filter_in += FilterTile)
//				{
//					cg::thread_block_tile<FilterTile, cg::thread_block> sub_block = cg::tiled_partition<FilterTile>(cg::this_thread_block());
//
//					// loading input
//					PaddedIndexer_v3<InputTile> i_workspace_indexer(FilterTile, InputTile);
//					for (int i = sub_block.meta_group_rank(); i < InputTile; i += sub_block.meta_group_size())
//						for (int j = sub_block.thread_rank(); j < FilterTile; j += sub_block.size())
//						{
//							const int offset = spatial_offsets[i];
//							const int f = filter_in + j;
//
//							T tmp = 0.0f;
//							if (f < input_filters and offset != -1)
//								tmp = input[offset + f];
//							input_workspace[i_workspace_indexer.at(j, i)] = tmp;
//						}
//
//					// loading weights
//					Indexer<4> weights_indexer(output_filters, KernelSize, KernelSize, input_filters);
//					PaddedIndexer_v3<OutputTile> w_workspace_indexer(FilterTile, OutputTile);
//					for (int i = sub_block.meta_group_rank(); i < OutputTile; i += sub_block.meta_group_size())
//						for (int j = sub_block.thread_rank(); j < FilterTile; j += sub_block.size())
//						{
//							const int f_out = blockIdx.x * OutputTile + j;
//							const int f_in = filter_in + j;
//
//							T tmp = 0.0f;
//							if (f_out < output_filters and f_in < input_filters)
//								tmp = weights[weights_indexer.at(f_out, kernel_row, kernel_col, f_in)];
//							weights_workspace[w_workspace_indexer.at(j, f_out)] = tmp;
//						}
//					__syncthreads();
//
//					// calculate dot products
//					const T *ptr_input = input_workspace;
//					const T *ptr_weights = weights_workspace;
//					for (int k = 0; k < FilterTile; k++)
//					{
//						const float4 lhs0 = *reinterpret_cast<const float4*>(ptr_input + 4 * (sub_block.meta_group_rank()));
//						const float4 lhs1 = *reinterpret_cast<const float4*>(ptr_input + 4 * (sub_block.meta_group_rank() + InputTile / 2));
//
//						const float4 rhs0 = one_x4(); //  = *reinterpret_cast<const float4*>(ptr_weights + 4 * (sub_block.thread_rank()));
//						const float4 rhs1 = one_x4(); //  = *reinterpret_cast<const float4*>(ptr_weights + 4 * (sub_block.thread_rank() + OutputTile / 2));
//
////						acc00.fma(lhs0, rhs0);
////						acc01.fma(lhs0, rhs1);
////						acc10.fma(lhs1, rhs0);
////						acc11.fma(lhs1, rhs1);
//						ptr_input += InputTile + 4;
//						ptr_weights += OutputTile + 4;
//					}
//					__syncthreads();
//				}
//			}
//
////		acc00.add(acc01);
////		acc00.add(acc10);
////		acc00.add(acc11);
//
//		for (int i = threadIdx.x; i < InputTile; i += blockDim.x)
//		{
//			const Indexer<4> indexer(batch_size, height, width, output_filters);
//			const int main_index = blockIdx.y * InputTile + i;
//			const int b = main_index / (height * width);
//			const int h = (main_index / width) % height;
//			const int w = main_index % width;
//			if (h < height and w < width and b < batch_size)
//				spatial_offsets[i] = indexer.at(b, h, w, 0);
//			else
//				spatial_offsets[i] = -1;
//		}
//		__syncthreads();
//
//		constexpr int HalfInputTile = InputTile / 2;
//		constexpr int HalfOutputTile = OutputTile / 2;
//
////		cg::thread_block_tile<FilterTile, cg::thread_block> sub_block = cg::tiled_partition<FilterTile>(cg::this_thread_block());
////		for (int i = 0; i < 2; i++)
////			for (int j = 0; j < 2; j++)
////				for (int row = 0; row < 4; row++)
////				{
////					const Indexer<2> indexer(HalfInputTile / 4, HalfOutputTile);
////
////					*reinterpret_cast<float4*>(workspace + indexer.at(sub_block.meta_group_rank(), sub_block.thread_rank() * 4)) = acc00[row];
////					__syncthreads();
////
////					const int x = i * HalfInputTile + sub_block.meta_group_rank() * 4 + row;
////					for (int fy = sub_block.thread_rank(); fy < HalfOutputTile; fy += sub_block.num_threads())
////					{
////						const int y = blockIdx.x * OutputTile + j * HalfOutputTile + fy;
////						if (spatial_offsets[x] != -1 and fy < output_filters)
////							output[spatial_offsets[x] + fy] = workspace[indexer.at(sub_block.meta_group_rank(), fy)];
////					}
////					__syncthreads();
////				}
//	}
//
//	template<int InputSpatialTile, int InputFiltersTile, int KernelSize, int OutputFiltersTile, typename T>
//	__launch_bounds__(256, 2)
//	__global__ void kernel_implicit_gemm(const T *input, const T *weights, T *output, int batch_size, int height, int width, int input_filters,
//			int output_filters)
//	{
//		assert(blockDim.x == 256);
//		constexpr int Padding = KernelSize / 2;
//		constexpr int TotalTileSize = InputSpatialTile + 2 * Padding;
//
//		__shared__ T input_workspace[square(TotalTileSize) * InputFiltersTile * vector_length<T>()];
//		__shared__ T weights_workspace[OutputFiltersTile * KernelSize * InputFiltersTile * vector_length<T>()]; // storage only for one row of a kernel
//
//		const int2 tile_layout = get_number_of_tiles(height, width, InputSpatialTile);
//		const int tile_index_h = blockIdx.y / tile_layout.x;
//		const int tile_index_w = blockIdx.y % tile_layout.x;
//
//		Tile<Vector<T>, 8, OutputFiltersTile / 16> accumulator;
//		accumulator.fill(vector_zero<T>());
//
//		for (int in = 0; in < input_filters; in += InputFiltersTile)
//		{ // loop over dot product length (input filters)
//
//			const PaddedIndexer input_workspace_indexer(TotalTileSize, TotalTileSize, InputFiltersTile);
//			TensorWrapper<3, T, PaddedIndexer> input_workspace_wrapper(input_workspace, input_workspace_indexer);
//
//			{ // load input data
//				cg::thread_block_tile<8, cg::thread_block> sub_block = cg::tiled_partition<8>(cg::this_thread_block());
//
//				ConstTensorWrapper<4, T> input_wrapper(input, batch_size, height, width, input_filters);
//
//				for (int in_tile_idx = sub_block.meta_group_rank(); in_tile_idx < square(TotalTileSize); in_tile_idx += sub_block.meta_group_size())
//				{
//					int f = sub_block.thread_rank();
//					const int in_tile_h = in_tile_idx / TotalTileSize;
//					const int in_tile_w = in_tile_idx % TotalTileSize;
//					const int h = InputSpatialTile * tile_index_h - Padding + in_tile_h;
//					const int w = InputSpatialTile * tile_index_w - Padding + in_tile_w;
//
//					Vector<T> tmp = vector_one<T>();
////					if (0 <= h and h < height and 0 <= w and w < width and (in + f) < input_filters)
////						tmp = input_wrapper.load(blockIdx.z, h, w, in + f);
////					input_workspace_wrapper.store(tmp, in_tile_h, in_tile_w, f);
//				}
//			}
//			TensorWrapper<3, T> weights_workspace_wrapper(weights_workspace, OutputFiltersTile, KernelSize, InputFiltersTile);
////			{ // load weights
////				ConstTensorWrapper<3, T> weights_wrapper(weights, output_filters, square(KernelSize), input_filters);
////				for (int out_filter_idx = tmp_thread_idx.y; out_filter_idx < OutputFiltersTile; out_filter_idx += 32)
////					for (int k = 0; k < square(KernelSize); k++) // loop over kernel elements
////						for (int f = tmp_thread_idx.x; f < InputFiltersTile; f += 8)
////						{
////							const int out_f = blockIdx.x * OutputFiltersTile + out_filter_idx;
////							const int in_f = in + f;
////
////							Vector<T> tmp = vector_zero<T>();
////							if (in_f < input_filters and out_f < output_filters)
////								tmp = weights_wrapper.load(out_f, k, in_f);
////							weights_workspace_wrapper.store(tmp, f, k, out_filter_idx);
////						}
////			}
//
////			__syncthreads();
//
//			cg::thread_block_tile<32, cg::thread_block> sub_block = cg::tiled_partition<32>(cg::this_thread_block());
//			for (int kernel_row = 0; kernel_row < KernelSize; kernel_row++)
//			{
//				for (int kernel_col = 0; kernel_col < KernelSize; kernel_col++)
//				{
//					for (int k = 0; k < InputFiltersTile; k++)
//					{
//						int offset = input_workspace_wrapper.indexer.at(kernel_row + sub_block.meta_group_rank(), kernel_col, k);
//						int stride = InputFiltersTile;
//						Line<Vector<T>, 8> lhs;
//						lhs.x0.load(input_workspace + offset + 0 * stride);
//						lhs.x1.load(input_workspace + offset + 1 * stride);
//						lhs.x2.load(input_workspace + offset + 2 * stride);
//						lhs.x3.load(input_workspace + offset + 3 * stride);
//						lhs.x4.load(input_workspace + offset + 4 * stride);
//						lhs.x5.load(input_workspace + offset + 5 * stride);
//						lhs.x6.load(input_workspace + offset + 6 * stride);
//						lhs.x7.load(input_workspace + offset + 7 * stride);
//
////						const int h = kernel_row + sub_block.meta_group_rank();
////						const int w = kernel_col;
////						Line<Vector<T>, 8> lhs;
////						lhs.x0 = input_workspace_wrapper.load(h, w + 0, k);
////						lhs.x1 = input_workspace_wrapper.load(h, w + 1, k);
////						lhs.x2 = input_workspace_wrapper.load(h, w + 2, k);
////						lhs.x3 = input_workspace_wrapper.load(h, w + 3, k);
////						lhs.x4 = input_workspace_wrapper.load(h, w + 4, k);
////						lhs.x5 = input_workspace_wrapper.load(h, w + 5, k);
////						lhs.x6 = input_workspace_wrapper.load(h, w + 6, k);
////						lhs.x7 = input_workspace_wrapper.load(h, w + 7, k);
//
//						offset = weights_workspace_wrapper.indexer.at(k, kernel_row * KernelSize + kernel_col, sub_block.thread_rank());
//						stride = 16;
//						Line<Vector<T>, accumulator.columns()> rhs;
////						reinterpret_cast<float2*>(&rhs)[0] = reinterpret_cast<float2*>(weights_workspace)[4 * (kernel_row * KernelSize + kernel_col)
////								+ k];
////						reinterpret_cast<float2*>(&rhs)[1] = reinterpret_cast<float2*>(weights_workspace)[4 * (kernel_row * KernelSize + kernel_col)
////								+ k + 1];
//						rhs.x0.load(weights_workspace + offset + 0 * stride);
//						rhs.x1.load(weights_workspace + offset + 1 * stride);
//						rhs.x2.load(weights_workspace + offset + 2 * stride);
//						rhs.x3.load(weights_workspace + offset + 3 * stride);
//
//						rhs.x4.load(weights_workspace + offset + 4 * stride);
//						rhs.x5.load(weights_workspace + offset + 5 * stride);
//						rhs.x6.load(weights_workspace + offset + 6 * stride);
//						rhs.x7.load(weights_workspace + offset + 7 * stride);
//
////						for (int i = 0; i < rhs.size(); i++)
////							rhs[i] = vector_one<T>(); // weights_workspace_wrapper.load(k, kernel_row * KernelSize + kernel_col, i * 16 + tmp_thread_idx.x);
//
//						outer_product(accumulator, lhs, rhs);
//					}
//				}
////				__syncthreads();
//			}
//
//		}
//
//		cg::thread_block_tile<32, cg::thread_block> sub_block = cg::tiled_partition<32>(cg::this_thread_block());
//		TensorWrapper<4, T> output_wrapper(output, batch_size, height, width, output_filters);
//		for (int r = 0; r < accumulator.rows(); r++)
//			for (int c = 0; c < accumulator.columns(); c++)
//			{
//				const int h = InputSpatialTile * tile_index_h + sub_block.meta_group_rank();
//				const int w = InputSpatialTile * tile_index_w + r;
//				const int out_f = blockIdx.x * OutputFiltersTile + c * 32 + sub_block.thread_rank();
//
//				if (h < height and w < width and out_f < output_filters)
//					output_wrapper.store(accumulator.at(r, c), blockIdx.z, h, w, out_f);
//			}
//	}
//
//	template<int InputSpatialTile, int InputFiltersTile, int KernelSize, int OutputFiltersTile, typename T>
//	__launch_bounds__(256, 2)
//	__global__ void kernel_implicit_gemm_conv(const T *input, const T *weights, T *output, int batch_size, int height, int width, int input_filters,
//			int output_filters)
//	{
//		assert(blockDim.x == 256);
//		constexpr int Padding = KernelSize / 2;
//		constexpr int TotalTileSize = InputSpatialTile + 2 * Padding;
//
//		__shared__ T input_workspace[square(TotalTileSize) * InputFiltersTile];
//		__shared__ T weights_workspace[OutputFiltersTile * square(KernelSize) * InputFiltersTile];
//
//		const int2 tile_layout = get_number_of_tiles(height, width, InputSpatialTile);
//		const int tile_index_h = blockIdx.y / tile_layout.x;
//		const int tile_index_w = blockIdx.y % tile_layout.x;
//
//		Tile<T, 4, 4> accumulator;
//
//		const int vertical_thread_idx = threadIdx.x / 16;
//		const int horizontal_thread_idx = threadIdx.x % 16;
//		for (int in = 0; in < input_filters; in += InputFiltersTile)
//		{ // loop over dot product length (input filters)
//
//			// load input data
//			ConstTensorWrapper<4, T> input_wrapper(input, batch_size, height, width, input_filters);
//			for (int in_tile_idx = vertical_thread_idx; in_tile_idx < square(TotalTileSize); in_tile_idx += 16)
//				for (int f = horizontal_thread_idx; f < InputFiltersTile; f += 16)
//				{
//					const int in_tile_h = in_tile_idx / TotalTileSize;
//					const int in_tile_w = in_tile_idx % TotalTileSize;
//					const int h = InputSpatialTile * tile_index_h - Padding + in_tile_h;
//					const int w = InputSpatialTile * tile_index_w - Padding + in_tile_w;
//
//					const int workspace_idx = (in_tile_h * TotalTileSize + in_tile_w) * InputFiltersTile + f;
//					if (0 <= h and h < height and 0 <= w and w < width and (in + f) < input_filters)
//					{
//						const int input_idx = ((blockIdx.z * height + h) * width + w) * input_filters + in + f;
//						input_workspace[workspace_idx] = input[input_idx];
//					}
//					else
//						input_workspace[workspace_idx] = 0.0f;
//				}
//
//			// load weights
//			for (int out_filter_idx = vertical_thread_idx; out_filter_idx < OutputFiltersTile; out_filter_idx += 16)
//				for (int k = 0; k < square(KernelSize); k++) // loop over kernel elements
//					for (int f = horizontal_thread_idx; f < InputFiltersTile; f += 16)
//					{
//						const int workspace_idx = (out_filter_idx * square(KernelSize) + k) * InputFiltersTile + f;
//						if ((in + f) < input_filters and (blockIdx.x * OutputFiltersTile + out_filter_idx) < output_filters)
//						{
//							const int weights_idx = ((blockIdx.x + out_filter_idx) * square(KernelSize) + k) * input_filters + in + f;
//							weights_workspace[workspace_idx] = input[weights_idx];
//						}
//						else
//							weights_workspace[workspace_idx] = 0.0f;
//					}
//			__syncthreads();
//
//			constexpr int HalfTileSize = InputSpatialTile / 2;
//			for (int kernel_h = 0; kernel_h < KernelSize; kernel_h++)
//				for (int kernel_w = 0; kernel_w < KernelSize; kernel_w++)
//					for (int f = 0; f < InputFiltersTile; f++)
//					{
//						int sub_tile_h = vertical_thread_idx / 4;
//						int sub_tile_w = vertical_thread_idx % 4;
//						int offset = ((kernel_h + sub_tile_h) * InputSpatialTile + kernel_w + sub_tile_w) * InputFiltersTile + f;
//						const float inp00 = input_workspace[offset];
//						const float inp01 = input_workspace[offset + HalfTileSize * InputFiltersTile];
//						const float inp10 = input_workspace[offset + HalfTileSize * InputSpatialTile * InputFiltersTile];
//						const float inp11 = input_workspace[offset + HalfTileSize * (1 + InputSpatialTile) * InputFiltersTile];
//
//						// OutputFiltersTile * square(KernelSize) * InputFiltersTile
//						offset = ((horizontal_thread_idx * KernelSize + kernel_h) * KernelSize + kernel_w) * InputFiltersTile + f;
//						const int stride = square(KernelSize) * InputFiltersTile;
//						const float w0 = weights_workspace[offset + 0 * stride];
//						const float w1 = 1.0f; // weights_workspace[offset + 1 * stride];
//						const float w2 = 1.0f; // weights_workspace[offset + 2 * stride];
//						const float w3 = 1.0f; // weights_workspace[offset + 3 * stride];
//
////						output_tile[0][0] += inp00 * w0;
////						output_tile[0][1] += inp00 * w1;
////						output_tile[0][2] += inp00 * w2;
////						output_tile[0][3] += inp00 * w3;
////
////						output_tile[1][0] += inp01 * w0;
////						output_tile[1][1] += inp01 * w1;
////						output_tile[1][2] += inp01 * w2;
////						output_tile[1][3] += inp01 * w3;
////
////						output_tile[2][0] += inp10 * w0;
////						output_tile[2][1] += inp10 * w1;
////						output_tile[2][2] += inp10 * w2;
////						output_tile[2][3] += inp10 * w3;
////
////						output_tile[3][0] += inp11 * w0;
////						output_tile[3][1] += inp11 * w1;
////						output_tile[3][2] += inp11 * w2;
////						output_tile[3][3] += inp11 * w3;
//					}
//			__syncthreads();
//		}
//
////		const int sub_tile_h = vertical_thread_idx / 4;
////		const int sub_tile_w = vertical_thread_idx % 4;
////		for (int i = 0; i < 4; i++)
////		{
////			for (int k = 0; k < 4; k++)
////				input_workspace[(sub_tile_h * 4 + sub_tile_w) * OutputFiltersTile + k * 16 + horizontal_thread_idx] = output_tile[i][k];
////
////			const int store_tile_h = threadIdx.x / 64;
////			const int store_tile_w = threadIdx.x % 64;
////
////			const int in_tile_h = (i / 2) * 4 + sub_tile_h;
////			const int in_tile_w = (i % 2) * 4 + sub_tile_w;
////			const int h = InputSpatialTile * tile_index_h + in_tile_h;
////			const int w = InputSpatialTile * tile_index_w + in_tile_w;
////
////			const int output_filter_idx = blockIdx.x * OutputFiltersTile + store_tile_w;
//////			if (0 <= h and h < height and 0 <= w and w < width)// and output_filter_idx < output_filters)
////			{
////				const int output_idx = ((blockIdx.z * height + h) * width + w) * output_filters + output_filter_idx;
////				output[output_idx] = input_workspace[store_tile_h * 64 + store_tile_w];
////			}
////		}
//	}
//
//	int get_output_filters_tile(int kernel_size)
//	{
//		switch (kernel_size)
//		{
//			case 1:
//				return 128;
//			case 3:
//				return 128;
//			case 5:
//				return 128;
//		}
//		return 0;
//	}
}

namespace ml
{
#ifndef USE_CUDNN
	void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
//		const int batch_size = input_shape.dim[0];
//		const int height = input_shape.dim[1];
//		const int width = input_shape.dim[2];
//		const int input_filters = input_shape.dim[3];
//		const int output_filters = weights_shape.dim[0];
//
//		const int filters_tile = get_output_filters_tile(weights_shape.dim[1]);
//
//		const int tiles_h = (height + 7) / 8;
//		const int tiles_w = (width + 7) / 8;
//		const int tiles_out = (output_filters + filters_tile - 1) / filters_tile;
////		dim3 gridSize(tiles_out, tiles_h * tiles_w, batch_size);
//		cudaStream_t stream = cuda::Context::getStream(context);
//
//		dim3 blockSize(256);
//		dim3 gridSize(tiles_out, (height * width * batch_size + 127) / 128);
//		kernel_implicit_gemm_v4<3, 128, 128, 32> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(input), getPointer<float>(weights),
//				getPointer<float>(output), batch_size, height, width, input_filters, output_filters);
////		kernel_implicit_gemm<8, 8, 3, 128> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(input), getPointer<float>(weights),
////				getPointer<float>(output), batch_size, height, width, input_filters, output_filters);
//		assert(cudaGetLastError() == cudaSuccess);
	}
#endif
} /* namespace ml */

