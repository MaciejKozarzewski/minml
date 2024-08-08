/*
 * gemm_runtime.hpp
 *
 *  Created on: May 18, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_GEMM_RUNTIME_HPP_
#define BACKEND_CPU_KERNELS_GEMM_GEMM_RUNTIME_HPP_

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <functional>
#include <iostream>

namespace ml
{
	struct TileDimensions
	{
			int M, N, K;
			std::string toString() const
			{
				return std::to_string(M) + " " + std::to_string(N) + " " + std::to_string(K);
			}
	};

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
			void create(cpu::WorkspaceAllocator &allocator, mlDataType_t dtype, Size2D max_size, size_t alignment = 4096) noexcept
			{
				m_max_size = max_size;
				const int fragment_size = size_of(dtype) * max_size.rows * max_size.columns;
				for (int i = 0; i < size(); i++)
					m_data[i] = Fragment(allocator.get(fragment_size, alignment), dtype, max_size.columns);
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

	static Matrix create_matrix(const void *ptr, mlDataType_t dtype, mlShape_t shape) noexcept
	{
		switch (shape.rank)
		{
			case 1:
				return Matrix(ptr, dtype, 1, shape.dim[0], 0);
			case 2:
				return Matrix(ptr, dtype, shape.dim[0], shape.dim[1], shape.dim[1]);
			case 3:
				return Matrix(ptr, dtype, shape.dim[1], shape.dim[2], shape.dim[2]);
			default:
				return Matrix();
		}
	}
	static MatrixOp convert_op(char op) noexcept
	{
		return is_transpose(op) ? MatrixOp::TRANSPOSE : MatrixOp::NORMAL;
	}
	static MatrixOp invert_op(MatrixOp op) noexcept
	{
		return (op == MatrixOp::NORMAL) ? MatrixOp::TRANSPOSE : MatrixOp::NORMAL;
	}

	struct TypeConfiguration
	{
			mlDataType_t matrix_d_dtype;
			mlDataType_t matrix_a_dtype;
			mlDataType_t matrix_b_dtype;
			mlDataType_t matrix_c_dtype;
			mlDataType_t compute_dtype;
	};

	class PerfEstimator
	{
			float m_a = 0.0f, m_b = 0.0f;
		public:
			PerfEstimator() noexcept = default;
			PerfEstimator(float a, float b) noexcept :
					m_a(a),
					m_b(b)
			{
			}
			float operator()(const TileDimensions &total_size, const TileDimensions &inner_tile) const noexcept
			{
				const int m_fragments = (total_size.M + inner_tile.M - 1) / inner_tile.M;
				const int n_fragments = (total_size.N + inner_tile.N - 1) / inner_tile.N;
				const double coverage = static_cast<double>(total_size.M * total_size.N) / (m_fragments * n_fragments * inner_tile.M * inner_tile.N);
				const double gflops = m_a * total_size.K / (total_size.K + m_b);
				return gflops * coverage;
			}
	};

	class GemmRuntime
	{
			Matrix matrix_A, matrix_B, matrix_C, matrix_D, bias;
			float alpha = 1.0f, beta = 0.0f;
			MatrixOp op_A = MatrixOp::NORMAL, op_B = MatrixOp::NORMAL;
			bool use_relu = false;

			TileDimensions outer_tile, total_size;

			ArrayOfFragments A_fragments, B_fragments, bias_fragments;
			Fragment edge_C_fragment, edge_D_fragment;

		public:
			using packing_function = std::function<void(Fragment&, const Matrix&, const Position2D&, MatrixOp)>;
			using unpacking_function = std::function<void(Matrix&, const Position2D&,const Fragment&)>;
			using gemm_function = std::function<void(Fragment &, const void *, const Fragment &, const Fragment &, const void *,
					const Fragment &, const Fragment &, bool)>;

			TypeConfiguration type_configuration;
			TileDimensions inner_tile;
			gemm_function gemm_kernel;
			packing_function a_packing, b_packing, c_packing, d_packing;
			packing_function edge_a_packing, edge_b_packing;
			unpacking_function d_unpacking;
			PerfEstimator perf_estimator;

			GemmRuntime() noexcept = default;
			bool can_work_with_types(const TypeConfiguration &tc) const noexcept
			{
				return type_configuration.matrix_a_dtype == tc.matrix_a_dtype and type_configuration.matrix_b_dtype == tc.matrix_b_dtype
						and type_configuration.matrix_c_dtype == tc.matrix_c_dtype and type_configuration.matrix_d_dtype == tc.matrix_d_dtype
						and type_configuration.compute_dtype == tc.compute_dtype;
			}
			float get_expected_gflops(int M, int N, int K) const noexcept
			{
				return perf_estimator( { M, N, K }, inner_tile);
			}
			void setMatrixA(const Matrix &mat, char op)
			{
				matrix_A = mat;
				op_A = invert_op(convert_op(op));
			}
			void setMatrixB(const Matrix &mat, char op)
			{
				matrix_B = mat;
				op_B = convert_op(op);
			}
			void setMatrixC(const Matrix &mat)
			{
				matrix_C = mat;
			}
			void setMatrixD(const Matrix &mat)
			{
				matrix_D = mat;
			}
			void setBias(const Matrix &mat)
			{
				bias = mat;
			}
			void setMatrixA(const void *ptr, mlShape_t shape, mlDataType_t dtype, char op)
			{
				setMatrixA(create_matrix(ptr, dtype, shape), op);
			}
			void setMatrixB(const void *ptr, mlShape_t shape, mlDataType_t dtype, char op)
			{
				setMatrixB(create_matrix(ptr, dtype, shape), op);
			}
			void setMatrixC(const void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				setMatrixC(create_matrix(ptr, dtype, shape));
			}
			void setMatrixD(void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				setMatrixD(create_matrix(ptr, dtype, shape));
			}
			void setScalingFactors(float alpha, float beta) noexcept
			{
				this->alpha = alpha;
				this->beta = beta;
			}
			void useRelu(bool b) noexcept
			{
				use_relu = b;
			}
			void setBias(const void *ptr, mlShape_t shape, mlDataType_t dtype)
			{
				setBias(create_matrix(ptr, dtype, shape));
			}
			void setup(mlContext_t context);
			void run();
		private:
			void find_tile_sizes();
			void create_fragments(mlContext_t context);
			void pack_fragment_A(Fragment &fragment, int m, int k);
			void pack_fragment_B(Fragment &fragment, int n, int k);
			void pack_fragment_C(Fragment &fragment, int m, int n);
			void pack_fragment_D(Fragment &fragment, int m, int n);
			void pack_fragment_bias(Fragment &fragment, int n);
			void unpack_fragment_D(Fragment &fragment, int m, int n);
	};

	GemmRuntime get_runtime(mlContext_t context, mlDataType_t dtype, char opA, mlShape_t shape_A, char opB, mlShape_t shape_B);
}

#endif /* BACKEND_CPU_KERNELS_GEMM_GEMM_RUNTIME_HPP_ */
