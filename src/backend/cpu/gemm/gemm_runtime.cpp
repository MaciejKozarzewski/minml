/*
 * gemm_runtime.cpp
 *
 *  Created on: May 18, 2023
 *      Author: Maciej Kozarzewski
 */

#include "gemm_runtime.hpp"

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

					ArrayOfFragments::Iterator B_frag_iter = B_fragments.get_iterator();
					for (int inner_n = outer_n; inner_n < std::min(total_size.N, outer_n + outer_tile.N); inner_n += inner_tile.N)
					{
						if (not B_frag_iter->is_packed())
							pack_fragment_B(*B_frag_iter, inner_n, outer_k);

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

							gemm_kernel(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment, use_relu and is_last_k_tile);
							unpack_fragment_D(D_fragment, inner_m, inner_n);
							A_frag_iter.advance();
						}
						B_frag_iter.advance();
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

		A_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(inner_tile.K, inner_tile.M));
		B_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(inner_tile.K, inner_tile.N));

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
	void GemmRuntime::unpack_fragment_D(Fragment &fragment, int m, int n)
	{
		const int rows_to_unpack = std::min(inner_tile.M, total_size.M - m);
		const int cols_to_unpack = std::min(inner_tile.N, total_size.N - n);

		const bool is_tile_full = (rows_to_unpack == inner_tile.M) and (cols_to_unpack == inner_tile.N);
		if (not is_tile_full)
			d_unpacking(matrix_D, Position2D(m, n), fragment);
	}

} /* namespace ml */

