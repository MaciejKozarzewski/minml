/*
 * gemm_runtime.cpp
 *
 *  Created on: May 18, 2023
 *      Author: Maciej Kozarzewski
 */

#include "gemm_runtime.hpp"

#include "../utils.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"
#include "gemm_kernels.hpp"
#include "gemm_runtime.hpp"

#include <limits>

namespace
{
	using namespace ml;
	Position2D get_position(int row, int col, MatrixOp op) noexcept
	{
		return (op == MatrixOp::NORMAL) ? Position2D(row, col) : Position2D(col, row);
	}

	TileDimensions get_outer_tile(const TileDimensions &inner_kernel) noexcept
	{
		const uint64_t cache_L1 = 32 * 1024; // [bytes]
		const uint64_t cache_L2 = 256 * 1024; // [bytes]
		const uint64_t cache_L3 = 6 * 1024 * 1024; // [bytes]

		return TileDimensions { 240, 192, inner_kernel.K };
	}
	int round_up(int x, int y) noexcept
	{
		return (x + y - 1) / y;
	}

	std::vector<GemmRuntime> get_sse2_gemm_runtime()
	{
		std::vector<GemmRuntime> result(1);
		// 4x8
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 4, 8, 256 };
		result[0].gemm_kernel = gemm_sse2_4x8_fp32;
		result[0].a_packing = pack_sse2_4xK_fp32;
		result[0].b_packing = pack_sse2_8xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(15.8, 14.2);

		return result;
	}
	std::vector<GemmRuntime> get_avx_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);
		// 10x8 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 10, 8, 256 };
		result[0].gemm_kernel = gemm_avx_10x8_fp32;
		result[0].a_packing = pack_avx_10xK_fp32;
		result[0].b_packing = pack_avx_8xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(31.6, 16.7);

		// 10x8 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 10, 8, 512 };
		result[1].gemm_kernel = gemm_avx_10x8_fp32_fp16;
		result[1].a_packing = pack_avx_10xK_fp16_fp32;
		result[1].b_packing = pack_avx_8xK_fp16_fp32;
		result[1].c_packing = pack_def_MxK_fp16;
		result[1].d_packing = pack_def_MxK_fp16;
		result[1].d_unpacking = unpack_def_MxK_fp16;
		result[1].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[1].perf_estimator = PerfEstimator(31.4, 14.6);

		return result;
	}
	std::vector<GemmRuntime> get_avx2_fma_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);

		// 12x8 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 12, 8, 1024 };
		result[0].gemm_kernel = gemm_avx2_fma_12x8_fp32;
		result[0].a_packing = pack_avx2_fma_12xK_fp32;
		result[0].b_packing = pack_avx_8xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(62.8, 23.4);

		// 12x8 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 12, 8, 1024 };
		result[1].gemm_kernel = gemm_avx2_fma_12x8_fp32_fp16;
		result[1].a_packing = pack_avx2_fma_12xK_fp16_fp32;
		result[1].b_packing = pack_avx_8xK_fp16_fp32;
		result[1].c_packing = pack_def_MxK_fp16;
		result[1].d_packing = pack_def_MxK_fp16;
		result[1].d_unpacking = unpack_def_MxK_fp16;
		result[1].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[1].perf_estimator = PerfEstimator(61.2, 25.5);

		return result;
	}
	std::vector<GemmRuntime> get_avx512f_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);

		// 24x16 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 24, 16, 1024 };
		result[0].gemm_kernel = gemm_avx512f_24x16_fp32;
		result[0].a_packing = pack_avx512f_24xK_fp32;
		result[0].b_packing = pack_avx512f_16xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(114.3, 42.8);

		// 24x16 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 24, 16, 1024 };
		result[1].gemm_kernel = gemm_avx512f_24x16_fp32_fp16;
		result[1].a_packing = pack_avx512f_24xK_fp16_fp32;
		result[1].b_packing = pack_avx512f_16xK_fp16_fp32;
		result[1].c_packing = pack_def_MxK_fp16;
		result[1].d_packing = pack_def_MxK_fp16;
		result[1].d_unpacking = unpack_def_MxK_fp16;
		result[1].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[1].perf_estimator = PerfEstimator(114.3, 42.8);

		return result;
	}

	template<typename T>
	void join_vectors(std::vector<T> &dst, const std::vector<T> &src)
	{
		dst.insert(dst.end(), src.begin(), src.end());
	}

	const std::vector<GemmRuntime>& get_gemm_runtime_table(mlContext_t context)
	{
		static const std::vector<GemmRuntime> runtime_table = [context]()
		{
			std::vector<GemmRuntime> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
			if (simd >= cpu::SimdLevel::AVX512F)
				join_vectors(result, get_avx512f_gemm_runtime());
			if (simd >= cpu::SimdLevel::AVX2)
				join_vectors(result, get_avx2_fma_gemm_runtime());
			if (simd >= cpu::SimdLevel::AVX)
				join_vectors(result, get_avx_gemm_runtime());
			if (simd >= cpu::SimdLevel::SSE2)
				join_vectors(result, get_sse2_gemm_runtime());
			return result;
		}();
		assert(runtime_table.size() > 0);
		return runtime_table;
	}

}

namespace ml
{

	void GemmRuntime::setup(mlContext_t context)
	{
		find_tile_sizes();
		create_fragments(context);
	}
	void GemmRuntime::run()
	{
		bool is_using_bias = (bias.data() != nullptr);
		Fragment C_fragment, D_fragment;

		for (int outer_m = 0; outer_m < total_size.M; outer_m += outer_tile.M)
			for (int outer_k = 0; outer_k < total_size.K; outer_k += outer_tile.K)
			{
				const float tmp_alpha = alpha;
				const float tmp_beta = (outer_k == 0) ? beta : 1.0f;
				const bool is_last_k_tile = (total_size.K - outer_k) <= outer_tile.K;

				A_fragments.reset();
				for (int outer_n = 0; outer_n < total_size.N; outer_n += outer_tile.N)
				{
					B_fragments.reset();
					bias_fragments.reset();

					ArrayOfFragments::Iterator B_frag_iter = B_fragments.get_iterator();
					ArrayOfFragments::Iterator bias_frag_iter = bias_fragments.get_iterator();
					for (int inner_n = outer_n; inner_n < std::min(total_size.N, outer_n + outer_tile.N); inner_n += inner_tile.N)
					{
						if (not B_frag_iter->is_packed())
							pack_fragment_B(*B_frag_iter, inner_n, outer_k);
						if (is_using_bias and is_last_k_tile and not bias_frag_iter->is_packed())
							pack_fragment_bias(*bias_frag_iter, inner_n);

						ArrayOfFragments::Iterator A_frag_iter = A_fragments.get_iterator();
						for (int inner_m = outer_m; inner_m < std::min(total_size.M, outer_m + outer_tile.M); inner_m += inner_tile.M)
						{
							if (not A_frag_iter->is_packed())
								pack_fragment_A(*A_frag_iter, inner_m, outer_k);

							pack_fragment_D(D_fragment, inner_m, inner_n);
							if (outer_k == 0)
								pack_fragment_C(C_fragment, inner_m, inner_n);
							else
								C_fragment = D_fragment;

							gemm_kernel(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment, *bias_frag_iter,
									use_relu and is_last_k_tile);
							unpack_fragment_D(D_fragment, inner_m, inner_n);
							A_frag_iter.advance();
						}
						B_frag_iter.advance();
						bias_frag_iter.advance();
					}
				}
			}
	}
	void GemmRuntime::find_tile_sizes()
	{
		assert(matrix_A.data() != nullptr && matrix_B.data() != nullptr && matrix_C.data() != nullptr && matrix_D.data() != nullptr); // all matrices have been set
		// TODO add better choice heuristics based on the problem dimensions

		const int M = matrix_D.rows();
		const int N = matrix_D.columns();
		const int K = (op_A == MatrixOp::NORMAL) ? matrix_A.rows() : matrix_A.columns();

		inner_tile.K = std::min(inner_tile.K, K);
		outer_tile = get_outer_tile(inner_tile);
		assert(outer_tile.M % inner_tile.M == 0);
		assert(outer_tile.N % inner_tile.N == 0);

		total_size = TileDimensions { M, N, K };
	}
	void GemmRuntime::create_fragments(mlContext_t context)
	{
		assert(inner_tile.M != 0 && inner_tile.N != 0 && inner_tile.K != 0); // tile sizes have been set
		const int num_A_fragments = round_up(std::min(total_size.M, outer_tile.M), inner_tile.M);
		const int num_B_fragments = round_up(std::min(total_size.N, outer_tile.N), inner_tile.N);

		cpu::WorkspaceAllocator workspace_allocator(context);
		A_fragments = ArrayOfFragments(workspace_allocator, num_A_fragments);
		B_fragments = ArrayOfFragments(workspace_allocator, num_B_fragments);
		bias_fragments = ArrayOfFragments(workspace_allocator, num_B_fragments);

		A_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(inner_tile.K, inner_tile.M));
		B_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(inner_tile.K, inner_tile.N));
		bias_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(1, inner_tile.N), 64);

		const Size2D tmp(inner_tile.M, inner_tile.N);
		const int fragment_size = size_of(matrix_D.dtype()) * tmp.rows * tmp.columns;
		edge_D_fragment = Fragment(workspace_allocator.get(fragment_size, 64), matrix_D.dtype(), matrix_D.stride());
		edge_C_fragment = Fragment(workspace_allocator.get(fragment_size, 64), matrix_C.dtype(), matrix_C.stride());
	}
	void GemmRuntime::pack_fragment_A(Fragment &fragment, int m, int k)
	{
		const int k_to_pack = std::min(inner_tile.K, total_size.K - k);
		const int m_to_pack = std::min(inner_tile.M, total_size.M - m);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, m_to_pack));

		const Position2D pos = get_position(k, m, op_A);
		if (fragment.columns() == fragment.stride())
			a_packing(fragment, matrix_A, pos, op_A);
		else
			edge_a_packing(fragment, matrix_A, pos, op_A);
	}
	void GemmRuntime::pack_fragment_B(Fragment &fragment, int n, int k)
	{
		const int k_to_pack = std::min(inner_tile.K, total_size.K - k);
		const int n_to_pack = std::min(inner_tile.N, total_size.N - n);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, n_to_pack));

		const Position2D pos = get_position(k, n, op_B);
		if (fragment.columns() == fragment.stride())
			b_packing(fragment, matrix_B, pos, op_B);
		else
			edge_b_packing(fragment, matrix_B, pos, op_B);
	}
	void GemmRuntime::pack_fragment_C(Fragment &fragment, int m, int n)
	{
		const int rows_to_pack = std::min(inner_tile.M, total_size.M - m);
		const int cols_to_pack = std::min(inner_tile.N, total_size.N - n);

		const bool is_tile_full = (rows_to_pack == inner_tile.M) and (cols_to_pack == inner_tile.N);
		if (is_tile_full)
		{
			const Size2D tmp(inner_tile.M, inner_tile.N);
			fragment = Fragment(matrix_C.pointer_at(m, n), matrix_C.dtype(), matrix_C.stride());
			fragment.set_size(tmp, matrix_C.stride());
		}
		else
		{
			fragment = edge_C_fragment;
			fragment.set_size(Size2D(rows_to_pack, cols_to_pack), inner_tile.N);
			c_packing(fragment, matrix_C, Position2D(m, n), MatrixOp::NORMAL);
		}
	}
	void GemmRuntime::pack_fragment_D(Fragment &fragment, int m, int n)
	{
		const int rows_to_pack = std::min(inner_tile.M, total_size.M - m);
		const int cols_to_pack = std::min(inner_tile.N, total_size.N - n);

		const bool is_tile_full = (rows_to_pack == inner_tile.M) and (cols_to_pack == inner_tile.N);
		if (is_tile_full)
		{
			const Size2D tmp(inner_tile.M, inner_tile.N);
			fragment = Fragment(matrix_D.pointer_at(m, n), matrix_D.dtype(), matrix_D.stride());
			fragment.set_size(tmp, matrix_D.stride());
		}
		else
		{
			fragment = edge_D_fragment;
			fragment.set_size(Size2D(rows_to_pack, cols_to_pack), inner_tile.N);
			d_packing(fragment, matrix_D, Position2D(m, n), MatrixOp::NORMAL);
		}
	}
	void GemmRuntime::pack_fragment_bias(Fragment &fragment, int n)
	{
		const int n_to_pack = std::min(inner_tile.N, total_size.N - n);
		fragment.mark_as_packed_with_size(Size2D(1, n_to_pack));

		const Position2D pos = get_position(1, n, MatrixOp::NORMAL);
		if (fragment.columns() == fragment.stride())
			b_packing(fragment, bias, pos, MatrixOp::NORMAL);
		else
			edge_b_packing(fragment, bias, pos, MatrixOp::NORMAL);
	}
	void GemmRuntime::unpack_fragment_D(Fragment &fragment, int m, int n)
	{
		const int rows_to_unpack = std::min(inner_tile.M, total_size.M - m);
		const int cols_to_unpack = std::min(inner_tile.N, total_size.N - n);

		const bool is_tile_full = (rows_to_unpack == inner_tile.M) and (cols_to_unpack == inner_tile.N);
		if (not is_tile_full)
			d_unpacking(matrix_D, Position2D(m, n), fragment);
	}

	GemmRuntime get_runtime(mlContext_t context, mlDataType_t dtype, char opA, mlShape_t shape_A, char opB, mlShape_t shape_B)
	{
		assert(shape_A.rank == 2 || shape_A.rank == 3);
		assert(shape_B.rank == 2 || shape_B.rank == 3);
		const TypeConfiguration tc { dtype, dtype, dtype, dtype, DTYPE_FLOAT32 };

		const int M = is_transpose(opA) ? shape_A.dim[shape_A.rank - 1] : shape_A.dim[shape_A.rank - 2];
		const int N = is_transpose(opB) ? shape_B.dim[shape_B.rank - 2] : shape_B.dim[shape_B.rank - 1];
		const int K = is_transpose(opA) ? shape_A.dim[shape_A.rank - 2] : shape_A.dim[shape_A.rank - 1];

		const std::vector<GemmRuntime> &table = get_gemm_runtime_table(context);
		GemmRuntime result;

		float max_gflops = std::numeric_limits<float>::lowest();
		for (auto iter = table.begin(); iter < table.end(); iter++)
		{
			if (iter->can_work_with_types(tc))
			{
				const float gflops = iter->get_expected_gflops(M, N, K);
				if (gflops > max_gflops)
				{
					result = *iter;
					max_gflops = gflops;
				}
			}
		}
		return result;
	}

} /* namespace ml */

