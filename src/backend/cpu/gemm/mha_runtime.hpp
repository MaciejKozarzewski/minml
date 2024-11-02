/*
 * mha_runtime.hpp
 *
 *  Created on: Sep 8, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_GEMM_MHA_RUNTIME_HPP_
#define BACKEND_CPU_GEMM_MHA_RUNTIME_HPP_

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

	struct MhaTypeConfiguration
	{
			mlDataType_t matrix_q_dtype;
			mlDataType_t matrix_k_dtype;
			mlDataType_t matrix_v_dtype;
			mlDataType_t bias_dtype;
			mlDataType_t compute_dtype;
	};

	class MhaRuntime
	{
			BatchedMatrix matrix_QKV, matrix_output, matrix_bias;

			TileDimensions total_size;

			ArrayOfFragments K_fragments, V_fragments;
			Fragment Q_fragment, temp_qk_fragment, edge_out_fragment, bias_fragment, softmax_sum_fragment;

			int batch_size = 0;
			int height = 0;
			int width = 0;
			int embedding = 0;
			int num_heads = 0;
			int bias_range = 0;
			int tokens = 0;
			int head_dim = 0;
			bool symmetric = false;

		public:
			using packing_function = std::function<void(Fragment&, const Matrix&, const Position2D&, MatrixOp)>;
			using unpacking_function = std::function<void(Matrix&, const Position2D&,const Fragment&)>;
			using mha1_function = std::function<void(Fragment&, const void*, const Fragment&, const Fragment&, const Fragment&, Fragment&)>;
			using mha2_function = std::function<void(Fragment &, const Fragment&, const Fragment &, const Fragment &, const void *,
					const Fragment &, const Fragment &, bool)>;

			MhaTypeConfiguration type_configuration;
			TileDimensions inner_tile;
			mha1_function softmax_qk_kernel;
			mha2_function mult_by_v_kernel;
			packing_function q_packing, kv_packing;
			PerfEstimator perf_estimator;

			MhaRuntime() noexcept = default;
			bool can_work_with_types(const MhaTypeConfiguration &tc) const noexcept
			{
				return type_configuration.matrix_q_dtype == tc.matrix_q_dtype and type_configuration.matrix_k_dtype == tc.matrix_k_dtype
						and type_configuration.matrix_v_dtype == tc.matrix_v_dtype and type_configuration.bias_dtype == tc.bias_dtype
						and type_configuration.compute_dtype == tc.compute_dtype;
			}
			float get_expected_gflops(int tokens, int head_dim) const noexcept
			{
				return perf_estimator( { tokens, tokens, head_dim }, inner_tile);
			}
			void setInput(const void *ptr, mlShape_t shape, mlDataType_t dtype, int num_heads, bool symmetric)
			{
				assert(shape.rank == 4);
				this->num_heads = num_heads;
				batch_size = shape.dim[0];
				height = shape.dim[1];
				width = shape.dim[2];
				assert(shape.dim[3] % (3 - symmetric) == 0);
				embedding = shape.dim[3] / (3 - symmetric);
				assert(embedding % num_heads == 0);
				this->symmetric = symmetric;
				matrix_QKV = BatchedMatrix(ptr, dtype, shape.dim[0], shape.dim[1] * shape.dim[2], shape.dim[3], shape.dim[3]);
			}
			void setOutput(void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				assert(shape.rank == 4);
				matrix_output = BatchedMatrix(ptr, dtype, shape.dim[0], shape.dim[1] * shape.dim[2], shape.dim[3], shape.dim[3]);
			}
			void setBias(const void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				if (ptr != nullptr and shape.rank == 3)
				{
					bias_range = (shape.dim[1] - 1) / 2;
					matrix_bias = BatchedMatrix(ptr, dtype, shape.dim[0], shape.dim[1], shape.dim[2], shape.dim[2]);
				}
				else
				{
					bias_range = 0;
					matrix_bias = BatchedMatrix();
				}
			}
			void setup(mlContext_t context);
			void run();
		private:
			void create_fragments(mlContext_t context);
			void pack_fragment_Q(const Matrix &Q, Fragment &fragment, int m, int k);
			void pack_fragment_K(const Matrix &K, Fragment &fragment, int n, int k);
			void pack_fragment_V(const Matrix &V, Fragment &fragment, int n, int k);
			void pack_fragment_out(Matrix &out, Fragment &fragment, int m, int n);
			void pack_fragment_bias(int head_idx);
			void unpack_fragment_out(Matrix &out, Fragment &fragment, int m, int n);
	};

	MhaRuntime get_mha_runtime(mlContext_t context, mlDataType_t dtype, int tokens, int head_dim);
}

#endif /* BACKEND_CPU_GEMM_MHA_RUNTIME_HPP_ */
