/*
 * gemm.cpp
 *
 *  Created on: May 10, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "../../utils.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"
#include "gemm_kernels.hpp"

#include <algorithm>
#include <iostream>

namespace
{
	using namespace ml;
	Matrix create_matrix(const void *ptr, mlDataType_t dtype, mlShape_t shape) noexcept
	{
		switch (shape.rank)
		{
			case 1:
				return Matrix(ptr, dtype, 1, shape.dim[0], 0);
			case 2:
				return Matrix(ptr, dtype, shape.dim[0], shape.dim[1], shape.dim[1]);
			default:
				return Matrix();
		}
	}

	struct TileDimensions
	{
			int M, N, K;
	};

	TileDimensions get_micro_kernel_tile(mlContext_t context, mlDataType_t dtype) noexcept
	{
//		switch (cpu::Context::getSimdLevel(context))
//		{
//			default:
//			case cpu::SimdLevel::NONE:
//		return TileDimensions { 4, 4, 128 };
//			case cpu::SimdLevel::SSE:
//			case cpu::SimdLevel::SSE2:
//			case cpu::SimdLevel::SSE3:
//			case cpu::SimdLevel::SSSE3:
//			case cpu::SimdLevel::SSE41:
//			case cpu::SimdLevel::SSE42:
//				return TileDimensions { 8, 4, 256 };
//			case cpu::SimdLevel::AVX:
//				return TileDimensions { 8, 8, 256 };
//			case cpu::SimdLevel::AVX2:
		return TileDimensions { 6, 16, 256 };
//			case cpu::SimdLevel::AVX512F:
//			case cpu::SimdLevel::AVX512VL_BW_DQ:
//				return TileDimensions { 12, 32, 256 };
//		}
	}
	TileDimensions get_outer_tile(mlContext_t context, mlDataType_t dtype) noexcept
	{
		const TileDimensions micro_kernel = get_micro_kernel_tile(context, dtype);

		const uint64_t cache_L1 = 32 * 1024; // [bytes]
		const uint64_t cache_L2 = 256 * 1024; // [bytes]
		const uint64_t cache_L3 = 6 * 1024 * 1024; // [bytes]

		return TileDimensions { 192, 192, micro_kernel.K };
	}

	class ArrayOfFragments
	{
			Fragment *m_data = nullptr;
			size_t m_length = 0;
			Size2D m_max_size;
		public:
			class Iterator
			{
					friend class ArrayOfFragments;
					Fragment *m_data = nullptr;
					size_t m_index = 0;
					Iterator(Fragment *ptr) :
							m_data(ptr)
					{
					}
				public:
					void advance() noexcept
					{
						m_index++;
					}
					const Fragment& operator*() const noexcept
					{
						return m_data[m_index];
					}
					Fragment& operator*() noexcept
					{
						return m_data[m_index];
					}
					const Fragment* operator->() const noexcept
					{
						return m_data + m_index;
					}
					Fragment* operator->() noexcept
					{
						return m_data + m_index;
					}
			};
			ArrayOfFragments() noexcept = default;
			ArrayOfFragments(cpu::WorkspaceAllocator &allocator, size_t length) :
					m_data(reinterpret_cast<Fragment*>(allocator.get(length * sizeof(Fragment), sizeof(Fragment)))),
					m_length(length)
			{
				assert(m_data != nullptr);
			}
			int size() const noexcept
			{
				return m_length;
			}
			Iterator get_iterator() noexcept
			{
				return Iterator(m_data);
			}
			void create(cpu::WorkspaceAllocator &allocator, mlDataType_t dtype, Size2D max_size) noexcept
			{
				m_max_size = max_size;
				const int fragment_size = size_of(dtype) * max_size.rows * max_size.columns;
				for (int i = 0; i < size(); i++)
					m_data[i] = Fragment(allocator.get(fragment_size, 4096), dtype);
			}
			const Fragment& operator[](int index) const noexcept
			{
				assert(0 <= index && index < size());
				return m_data[index];
			}
			Fragment& operator[](int index) noexcept
			{
				assert(0 <= index && index < size());
				return m_data[index];
			}
			void reset() noexcept
			{
				for (int i = 0; i < size(); i++)
					m_data[i].set_size(m_max_size, m_max_size.columns);
			}
	};

	enum class Use
	{
		MATRIX_A,
		MATRIX_B,
		MATRIX_C,
		MATRIX_D
	};
//
//	template<Use U>
//	bool is_fragment_full(const Fragment &frag, const TileDimensions &innerTile) noexcept
//	{
//		switch (U)
//		{
//			case Use::MATRIX_A:
//				return frag.rows() == innerTile.K and frag.columns() == innerTile.M;
//			case Use::MATRIX_B:
//				return frag.rows() == innerTile.K and frag.columns() == innerTile.N;
//			case Use::MATRIX_C:
//			case Use::MATRIX_D:
//				return frag.rows() == innerTile.M and frag.columns() == innerTile.N;
//			default:
//				return false;
//		}
//	}

	MatrixOp convert_op(char op) noexcept
	{
		assert(op == 'n' || op == 't');
		return (op == 'n') ? MatrixOp::NORMAL : MatrixOp::TRANSPOSE;
	}
	MatrixOp invert_op(MatrixOp op) noexcept
	{
		return (op == MatrixOp::NORMAL) ? MatrixOp::TRANSPOSE : MatrixOp::NORMAL;
	}
	Position2D get_position(int row, int col, MatrixOp op) noexcept
	{
		return (op == MatrixOp::NORMAL) ? Position2D(row, col) : Position2D(col, row);
	}
	Size2D transform(int rows, int cols, MatrixOp op) noexcept
	{
		return (op == MatrixOp::NORMAL) ? Size2D(rows, cols) : Size2D(cols, rows);
	}
	bool is_transpose(MatrixOp op) noexcept
	{
		return op == MatrixOp::TRANSPOSE;
	}
	int round_up(int x, int y) noexcept
	{
		return (x + y - 1) / y;
	}

	template<typename T>
	void print_fragment(const Fragment &frag)
	{
		std::cout << "Printing fragment of size " << frag.rows() << "x" << frag.columns() << " with stride " << frag.stride() << '\n';
		for (int row = 0; row < frag.rows(); row++)
		{
			for (int col = 0; col < frag.columns(); col++)
				std::cout << frag.at<T>(row, col) << ' ';
			std::cout << '\n';
		}
	}
}

namespace ml
{
	struct GemmRuntime
	{
			mlContext_t context;
			cpu::WorkspaceAllocator workspace_allocator;
			const MatrixOp op_A, op_B;
			const mlDataType_t compute_type;
			Matrix matrix_A, matrix_B, matrix_C, matrix_D;
			float alpha = 1.0f, beta = 0.0f;

			TileDimensions inner_tile, outer_tile, total_size;

			ArrayOfFragments A_fragments, B_fragments;
			Fragment edge_C_fragment, edge_D_fragment;

			size_t micro_kernel_count = 0;
			size_t pack_a_count = 0;
			size_t pack_b_count = 0;
			size_t pack_c_count = 0;
			size_t pack_d_count = 0, unpack_d_count = 0;

			GemmRuntime(mlContext_t context, char opA, char opB, mlDataType_t computeType) :
					context(context),
					workspace_allocator(context),
					op_A(invert_op(convert_op(opA))),
					op_B(convert_op(opB)),
					compute_type(computeType)
			{
			}
			void setMatrixA(const void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				matrix_A = create_matrix(ptr, dtype, shape);
			}
			void setMatrixB(const void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				matrix_B = create_matrix(ptr, dtype, shape);
			}
			void setMatrixC(const void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				matrix_C = create_matrix(ptr, dtype, shape);
			}
			void setMatrixD(void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				matrix_D = create_matrix(ptr, dtype, shape);
			}
			void setScalingFactors(float alpha, float beta) noexcept
			{
				this->alpha = alpha;
				this->beta = beta;
			}
			void run()
			{
				find_tile_sizes();
				create_fragments();

				Fragment C_fragment, D_fragment;

				for (int outer_m = 0; outer_m < total_size.M; outer_m += outer_tile.M)
					for (int outer_k = 0; outer_k < total_size.K; outer_k += outer_tile.K)
					{
						const float tmp_alpha = alpha;
						const float tmp_beta = (outer_k == 0) ? beta : 1.0f;

						A_fragments.reset();
						for (int outer_n = 0; outer_n < total_size.N; outer_n += outer_tile.N)
						{
							B_fragments.reset();
//							std::cout << "----outer tile " << outer_m << " " << outer_n << " " << outer_k << '\n';

							ArrayOfFragments::Iterator B_frag_iter = B_fragments.get_iterator();
							for (int inner_n = outer_n; inner_n < std::min(total_size.N, outer_n + outer_tile.N); inner_n += inner_tile.N)
							{
								if (not B_frag_iter->is_packed())
									pack_fragment_B(*B_frag_iter, inner_n, outer_k);

								ArrayOfFragments::Iterator A_frag_iter = A_fragments.get_iterator();
								for (int inner_m = outer_m; inner_m < std::min(total_size.M, outer_m + outer_tile.M); inner_m += inner_tile.M)
								{
//									std::cout << "inner tile " << inner_m << " " << inner_n << '\n';
									if (not A_frag_iter->is_packed())
										pack_fragment_A(*A_frag_iter, inner_m, outer_k);

									pack_fragment_D(D_fragment, inner_m, inner_n);
									if (outer_k == 0)
										pack_fragment_C(C_fragment, inner_m, inner_n);
									else
										C_fragment = D_fragment;

//									gemm_def_MxN_fp32(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment);
//									gemm_sse2_8x4_fp32(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment);
//									gemm_avx_8x8_fp32(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment);
									gemm_avx2_fma_6x16_fp32(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment);
//									gemm_avx2_fma_5x16_fp32(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment);
									micro_kernel_count++;

									unpack_fragment_D(D_fragment, inner_m, inner_n);
									A_frag_iter.advance();
								}
								B_frag_iter.advance();
							}
						}
					}
//				std::cout << "pack_a_count = " << pack_a_count << '\n';
//				std::cout << "pack_b_count = " << pack_b_count << '\n';
//				std::cout << "pack_c_count = " << pack_c_count << '\n';
//				std::cout << "pack_d_count = " << pack_d_count << ", unpack_d_count = " << unpack_d_count << '\n';
//				std::cout << "micro_kernel_count = " << micro_kernel_count << '\n';
//				exit(0);
			}
		private:
			Size2D get_fragment_size(Use use) const noexcept
			{
				switch (use)
				{
					case Use::MATRIX_A:
						return Size2D(inner_tile.K, inner_tile.M);
					case Use::MATRIX_B:
						return Size2D(inner_tile.K, inner_tile.N);
					case Use::MATRIX_C:
					case Use::MATRIX_D:
						return Size2D(inner_tile.M, inner_tile.N);
					default:
						return Size2D();
				}
			}
			void find_tile_sizes()
			{
				assert(matrix_A.data() != nullptr && matrix_B.data() != nullptr && matrix_C.data() != nullptr && matrix_D.data() != nullptr); // all matrices have been set
				// TODO add better choice heuristics based on the problem dimensions
				inner_tile = get_micro_kernel_tile(context, compute_type);
				outer_tile = get_outer_tile(context, compute_type);
				assert(outer_tile.M % inner_tile.M == 0);
				assert(outer_tile.N % inner_tile.N == 0);

				const int M = matrix_D.rows();
				const int N = matrix_D.columns();
				const int K = (op_A == MatrixOp::NORMAL) ? matrix_A.rows() : matrix_A.columns();
				total_size = TileDimensions { M, N, K };
			}
			void create_fragments()
			{
				assert(inner_tile.M != 0 && inner_tile.N != 0 && inner_tile.K != 0); // tile sizes have been set
				const int num_A_fragments = round_up(std::min(total_size.M, outer_tile.M), inner_tile.M);
				const int num_B_fragments = round_up(std::min(total_size.N, outer_tile.N), inner_tile.N);

				A_fragments = ArrayOfFragments(workspace_allocator, num_A_fragments);
				B_fragments = ArrayOfFragments(workspace_allocator, num_B_fragments);

				A_fragments.create(workspace_allocator, compute_type, get_fragment_size(Use::MATRIX_A));
				B_fragments.create(workspace_allocator, compute_type, get_fragment_size(Use::MATRIX_B));

				const Size2D tmp = get_fragment_size(Use::MATRIX_D);
				const int fragment_size = size_of(matrix_D.dtype()) * tmp.rows * tmp.columns;
				edge_D_fragment = Fragment(workspace_allocator.get(fragment_size, 64), matrix_D.dtype());
				edge_C_fragment = Fragment(workspace_allocator.get(fragment_size, 64), matrix_C.dtype());
			}
			void pack_fragment_A(Fragment &fragment, int m, int k)
			{
				pack_a_count++;

				const int k_to_pack = std::min(inner_tile.K, total_size.K - k);
				const int m_to_pack = std::min(inner_tile.M, total_size.M - m);
				fragment.mark_as_packed_with_size(Size2D(k_to_pack, m_to_pack), fragment.stride());

				const Position2D pos = get_position(k, m, op_A);
//				pack_def_MxK_fp32(fragment, matrix_A, pos, op_A);
				pack_avx2_fma_6xK_fp32(fragment, matrix_A, pos, op_A);
//				print_fragment<float>(fragment);
//				exit(0);
			}
			void pack_fragment_B(Fragment &fragment, int n, int k)
			{
				pack_b_count++;

				const int k_to_pack = std::min(inner_tile.K, total_size.K - k);
				const int n_to_pack = std::min(inner_tile.N, total_size.N - n);
				fragment.mark_as_packed_with_size(Size2D(k_to_pack, n_to_pack), fragment.stride());

				const Position2D pos = get_position(k, n, op_B);
//				pack_def_MxK_fp32(fragment, matrix_B, pos, op_B);
				pack_avx2_fma_16xK_fp32(fragment, matrix_B, pos, op_B);
			}
			void pack_fragment_C(Fragment &fragment, int m, int n)
			{
				const int rows_to_pack = std::min(inner_tile.M, total_size.M - m);
				const int cols_to_pack = std::min(inner_tile.N, total_size.N - n);

				const bool is_tile_full = (rows_to_pack == inner_tile.M) and (cols_to_pack == inner_tile.N);
				if (is_tile_full)
				{
					const Size2D tmp(inner_tile.M, inner_tile.N);
					fragment = Fragment(matrix_C.pointer_at(m, n), matrix_C.dtype());
					fragment.set_size(tmp, matrix_C.stride());
				}
				else
				{
					pack_c_count++;
					fragment = edge_C_fragment;
					fragment.set_size(Size2D(rows_to_pack, cols_to_pack), inner_tile.N);
					pack_def_MxK_fp32(fragment, matrix_C, Position2D(m, n), MatrixOp::NORMAL);
				}
			}
			void pack_fragment_D(Fragment &fragment, int m, int n)
			{
				const int rows_to_pack = std::min(inner_tile.M, total_size.M - m);
				const int cols_to_pack = std::min(inner_tile.N, total_size.N - n);

				const bool is_tile_full = (rows_to_pack == inner_tile.M) and (cols_to_pack == inner_tile.N);
				if (is_tile_full)
				{
					const Size2D tmp(inner_tile.M, inner_tile.N);
					fragment = Fragment(matrix_D.pointer_at(m, n), matrix_D.dtype());
					fragment.set_size(tmp, matrix_D.stride());
				}
				else
				{
					pack_d_count++;
					fragment = edge_D_fragment;
					fragment.set_size(Size2D(rows_to_pack, cols_to_pack), inner_tile.N);
					pack_def_MxK_fp32(fragment, matrix_D, Position2D(m, n), MatrixOp::NORMAL);
				}
			}
			void unpack_fragment_D(Fragment &fragment, int m, int n)
			{
				const int rows_to_unpack = std::min(inner_tile.M, total_size.M - m);
				const int cols_to_unpack = std::min(inner_tile.N, total_size.N - n);

				const bool is_tile_full = (rows_to_unpack == inner_tile.M) and (cols_to_unpack == inner_tile.N);
				if (not is_tile_full)
				{
					unpack_d_count++;
					unpack_def_MxK_fp32(matrix_D, Position2D(m, n), fragment);
				}
			}
	};

	void cpu_gemm_v2(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C)
	{
		GemmRuntime tmp(context, opA, opB, DTYPE_FLOAT32);
		tmp.setMatrixA(A, shape_A, dtype);
		tmp.setMatrixB(B, shape_B, dtype);
		tmp.setMatrixC(C, shape_C, dtype);
		tmp.setMatrixD(D, shape_D, dtype);
		tmp.setScalingFactors(alpha, beta);
		tmp.run();

//		const MatrixOp op_A = invert_op(convert_op(opA));
//		const MatrixOp op_B = convert_op(opB);
//
//		const Matrix matrix_A = create_matrix(A, dtype, shape_A);
//		const Matrix matrix_B = create_matrix(B, dtype, shape_B);
//		const Matrix matrix_C = create_matrix(C, dtype, shape_C);
//		Matrix matrix_D = create_matrix(D, dtype, shape_D);
//
//		const TileDimensions inner_tile = get_micro_kernel_tile(context, dtype);
//		const TileDimensions outer_tile = get_outer_tile(context, dtype);
//		const TileDimensions total_size { matrix_D.rows(), matrix_D.columns(), (opA == 'n') ? matrix_A.columns() : matrix_A.rows() };
//
//		const int num_A_fragments = outer_tile.M / inner_tile.M;
//		const int num_B_fragments = outer_tile.N / inner_tile.N;
//
//		cpu::WorkspaceAllocator workspace_allocator(context);
//
//		ArrayOfFragments A_fragments(workspace_allocator, num_A_fragments);
//		ArrayOfFragments B_fragments(workspace_allocator, num_B_fragments);
//		ArrayOfFragments CD_fragments(workspace_allocator, 2); // storage only for edge output tiles (one for C and another for D)
//
//		A_fragments.initialize(workspace_allocator, dtype, inner_tile.K, inner_tile.M, inner_tile.M);
//		B_fragments.initialize(workspace_allocator, dtype, inner_tile.K, inner_tile.N, inner_tile.N);
//		CD_fragments.initialize(workspace_allocator, dtype, inner_tile.M, inner_tile.N, inner_tile.N);
//
//		for (int outer_n = 0; outer_n < total_size.N; outer_n += outer_tile.N)
//			for (int outer_k = 0; outer_k < total_size.K; outer_k += outer_tile.K)
//			{
//				const float tmp_alpha = alpha;
//				const float tmp_beta = (outer_k == 0) ? beta : 1.0f;
//
//				B_fragments.reset_flags();
//				for (int outer_m = 0; outer_m < total_size.M; outer_m += outer_tile.M)
//				{
//					std::cout << "tile " << outer_n << " " << outer_k << " " << outer_m << " : " << tmp_alpha << " " << tmp_beta << '\n';
//					A_fragments.reset_flags();
//					ArrayOfFragments::Iterator B_frag_iter = B_fragments.get_iterator();
//					for (int inner_n = outer_n; inner_n < std::min(total_size.N, outer_n + outer_tile.N); inner_n += inner_tile.N)
//					{
//						if (not B_frag_iter->is_packed())
//						{
//							const Position2D pos = get_position(outer_k, outer_n + inner_n, op_B);
//							pack_def_MxK_fp32(*B_frag_iter, matrix_B, pos, op_B);
//							B_frag_iter->mark_as_packed();
//						}
//
//						ArrayOfFragments::Iterator A_frag_iter = A_fragments.get_iterator();
//						for (int inner_m = outer_m; inner_m < std::min(total_size.M, outer_m + outer_tile.M); inner_m += inner_tile.M)
//						{
//							if (not A_frag_iter->is_packed())
//							{
//								const Position2D pos = get_position(outer_k, outer_m + inner_m, op_A);
//								pack_def_MxK_fp32(*A_frag_iter, matrix_A, pos, op_A);
//								A_frag_iter->mark_as_packed();
//							}
//
//							Fragment C_fragment;
//							Fragment D_fragment;
//
//							if (true)
//							{
//								D_fragment = Fragment(matrix_D.pointer_at(outer_m + inner_m, outer_n + inner_n), inner_tile.M, inner_tile.N,
//										inner_tile.N);
//
//								if (outer_k == 0)
//								{
//									C_fragment = Fragment(matrix_C.pointer_at(outer_m + inner_m, outer_n + inner_n), inner_tile.M, inner_tile.N,
//											inner_tile.N);
//								}
//								else
//									C_fragment = D_fragment;
//							}
//
//							gemm_def_MxN_fp32(D_fragment, &tmp_alpha, *A_frag_iter, *B_frag_iter, &tmp_beta, C_fragment);
//							A_frag_iter.advance();
//						}
//						B_frag_iter.advance();
//					}
//				}
//			}
	}
} /* namespace ml */

