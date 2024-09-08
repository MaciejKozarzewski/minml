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
			Matrix matrix_Q, matrix_K, matrix_V, bias;

			TileDimensions outer_tile, total_size;

			ArrayOfFragments Q_fragments, K_fragments, V_fragments, bias_fragments;
			Fragment edge_out_fragment, bias_copy;

		public:
			using packing_function = std::function<void(Fragment&, const Matrix&, const Position2D&, MatrixOp)>;
			using unpacking_function = std::function<void(Matrix&, const Position2D&,const Fragment&)>;
			using mha1_function = std::function<void(Fragment&, const void*, const Fragment&, const Fragment&, const Fragment&, Fragment&)>;
			using mha2_function = std::function<void(Fragment &, const void *, const Fragment &, const Fragment &, const void *,
					const Fragment &, const Fragment &, bool)>;

			MhaTypeConfiguration type_configuration;
			TileDimensions inner_tile;
			mha1_function softmax_qk_kernel;
			mha2_function mult_by_v_kernel;
			packing_function q_packing, k_packing, v_packing;
			packing_function edge_q_packing, edge_k_packing, edge_v_packing;
			unpacking_function out_unpacking;
			PerfEstimator perf_estimator;

			MhaRuntime() noexcept = default;
			bool can_work_with_types(const MhaTypeConfiguration &tc) const noexcept
			{
				return true;
			}
			float get_expected_gflops(int M, int N, int K) const noexcept
			{
				return perf_estimator( { M, N, K }, inner_tile);
			}
			void setMatrixQ(const Matrix &mat)
			{
				matrix_Q = mat;
			}
			void setMatrixK(const Matrix &mat)
			{
				matrix_Q = mat;
			}
			void setMatrixV(const Matrix &mat)
			{
				matrix_V = mat;
			}
			void setBias(const Matrix &mat)
			{
				bias = mat;
				bias_copy.set_size( Size2D(), 0);
			}
			void setup(mlContext_t context);
			void run();
		private:
			void find_tile_sizes();
			void create_fragments(mlContext_t context);
			void pack_fragment_Q(Fragment &fragment, int m, int k);
			void pack_fragment_K(Fragment &fragment, int n, int k);
			void pack_fragment_V(Fragment &fragment, int n, int k);
			void pack_fragment_out(Fragment &fragment, int m, int n);
			void pack_fragment_bias(Fragment &fragment, int m, int n);
			void unpack_fragment_out(Fragment &fragment, int m, int n);
	};

	MhaRuntime get_mha_runtime(mlContext_t context, mlDataType_t dtype, char opA, char opB, mlShape_t shape_A, mlShape_t shape_B);
	MhaRuntime get_mha_runtime(mlContext_t context, mlDataType_t dtype, char opA, char opB, int M, int N, int K);
}

#endif /* BACKEND_CPU_GEMM_MHA_RUNTIME_HPP_ */
