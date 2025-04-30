/*
 * fused_conv_block_runtime.hpp
 *
 *  Created on: Apr 16, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_GEMM_FUSED_CONV_BLOCK_RUNTIME_HPP_
#define BACKEND_CPU_GEMM_FUSED_CONV_BLOCK_RUNTIME_HPP_

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "gemm_runtime.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <functional>
#include <iostream>

namespace ml
{

	class FCBRuntime
	{
			Matrix input_matrix, output_matrix;
			Matrix dwconv_weights, dwconv_bias;
			Matrix first_conv_weights, first_conv_bias;
			Matrix second_conv_weights, second_conv_bias;

			Matrix dw_out_matrix;
			ArrayOfFragments w1_fragments, b1_fragments;
			ArrayOfFragments w2_fragments, b2_fragments;

			Fragment dwconv_output_fragment;
			Fragment first_conv_output_fragment;
			Fragment edge_output_fragment;

			int batch_size = 0;
			int height = 0;
			int width = 0;
			int channels = 0;
			int hidden_dim = 0;
			int dw_kernel_size = 0;
			int *indices = nullptr;

		public:
			using packing_function = std::function<void(Fragment&, const Matrix&, const Position2D&, MatrixOp)>;
			using unpacking_function = std::function<void(Matrix&, const Position2D&,const Fragment&)>;
			using dwconv_function = std::function<void(Matrix &, const Matrix &, const Matrix &, const Matrix &, const int *,
					void *)>;
			using conv1_function = std::function<void(Fragment&, const Fragment&, const Fragment&, const Fragment&)>;
			using conv2_function = std::function<void(Fragment &, const Fragment&, const Fragment &, const Fragment &, const void *,
					const Fragment &, const Fragment &, bool)>;

			mlDataType_t dtype = DTYPE_UNKNOWN;
			TileDimensions inner_tile;
			dwconv_function dwconv_kernel;
			conv1_function first_conv_kernel;
			conv2_function second_conv_kernel;
			packing_function input_packing, weights_packing;
			PerfEstimator perf_estimator;

			bool can_work_with_types(mlDataType_t dt) const noexcept
			{
				return dtype == dt;
			}
			float get_expected_gflops(int tokens, int head_dim) const noexcept
			{
				return perf_estimator( { tokens, tokens, head_dim }, inner_tile);
			}
			void setInput(const mlTensor_t &tensor)
			{
				assert(tensor.rank == 4);
				assert(tensor.dtype == dtype);
				batch_size = tensor.dim[0];
				height = tensor.dim[1];
				width = tensor.dim[2];
				channels = tensor.dim[3];
				input_matrix = Matrix(tensor.data, tensor.dtype, tensor.dim[0] * tensor.dim[1] * tensor.dim[2], tensor.dim[3], tensor.dim[3]);
			}
			void setOutput(const mlTensor_t &tensor)
			{
				assert(tensor.rank == 4);
				assert(tensor.dim[0] == batch_size);
				assert(tensor.dim[1] == height);
				assert(tensor.dim[2] == width);
				assert(tensor.dim[3] == channels);
				assert(tensor.dtype == dtype);
				output_matrix = Matrix(tensor.data, tensor.dtype, tensor.dim[0] * tensor.dim[1] * tensor.dim[2], tensor.dim[3], tensor.dim[3]);
			}
			void setDepthwiseConv(const mlTensor_t &weights, const mlTensor_t &bias)
			{
				assert(weights.rank == 3);
				assert(weights.dim[0] == weights.dim[1]); // square kernel
				assert(weights.dim[2] == channels);
				assert(weights.dtype == dtype);
				dw_kernel_size = weights.dim[0];
				dwconv_weights = Matrix(weights.data, weights.dtype, weights.dim[0] * weights.dim[1], weights.dim[2], weights.dim[2]);

				assert(bias.rank == 1);
				assert(bias.dim[0] == channels);
				assert(bias.dtype == dtype);
				dwconv_bias = Matrix(bias.data, bias.dtype, 1, bias.dim[0], 0);
			}
			void setFirstConv(const mlTensor_t &weights, const mlTensor_t &bias)
			{
				assert(weights.rank == 4);
				assert(weights.dim[1] == 1);
				assert(weights.dim[2] == 1);
				assert(weights.dim[3] == channels);
				assert(weights.dtype == dtype);
				hidden_dim = weights.dim[0];
				first_conv_weights = Matrix(weights.data, weights.dtype, weights.dim[0], weights.dim[3], weights.dim[3]);

				assert(bias.rank == 1);
				assert(bias.dim[0] == hidden_dim);
				assert(bias.dtype == dtype);
				first_conv_bias = Matrix(bias.data, bias.dtype, 1, bias.dim[0], 0);
			}
			void setSecondConv(const mlTensor_t &weights, const mlTensor_t &bias)
			{
				assert(weights.rank == 4);
				assert(weights.dim[0] == channels);
				assert(weights.dim[1] == 1);
				assert(weights.dim[2] == 1);
				assert(weights.dim[3] == hidden_dim);
				assert(weights.dtype == dtype);
				second_conv_weights = Matrix(weights.data, weights.dtype, weights.dim[0], weights.dim[3], weights.dim[3]);

				assert(bias.rank == 1);
				assert(bias.dim[0] == channels);
				assert(bias.dtype == dtype);
				second_conv_bias = Matrix(bias.data, bias.dtype, 1, bias.dim[0], 0);
			}
			void setup(mlContext_t context);
			void run();
		private:
			void create_fragments(mlContext_t context);
			void pack_fragment_input(Fragment &fragment, int m);
			void pack_fragment_weights(const Matrix &weights, Fragment &fragment, int n);
			void pack_fragment_out(Matrix &out, Fragment &fragment, int m, int n);
			void pack_fragment_bias(const Matrix &bias, Fragment &fragment, int n);
			void unpack_fragment_out(Fragment &fragment, int m, int n);
	};

	FCBRuntime get_fcb_runtime(mlContext_t context, mlDataType_t dtype);
}

#endif /* BACKEND_CPU_GEMM_FUSED_CONV_BLOCK_RUNTIME_HPP_ */
