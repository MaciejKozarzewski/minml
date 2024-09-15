/*
 * mha_runtime.cpp
 *
 *  Created on: Sep 12, 2024
 *      Author: Maciej Kozarzewski
 */

#include "mha_runtime.hpp"

#include "../utils.hpp"
#include "../fp16.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"
#include "gemm_kernels.hpp"
#include "gemm_runtime.hpp"

#include <limits>
#include <cmath>
#include <cstring>

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
	int round_to_multiple_of(int x, int y) noexcept
	{
		const int r = x % y;
		return (r == 0) ? x : (x + y - r);
	}
	int divide_rounding_up(int x, int y) noexcept
	{
		return (x + y - 1) / y;
	}

	std::vector<MhaRuntime> get_sse2_mha_runtime()
	{
		std::vector<MhaRuntime> result(1);
		// 4x8
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 4, 8, 256 };
		result[0].softmax_qk_kernel = mha_qk_sse2_4x8;
		result[0].mult_by_v_kernel = gemm_sse2_4x8;
		result[0].q_packing = pack_sse2_4xK;
		result[0].kv_packing = pack_sse2_8xK;
		result[0].perf_estimator = PerfEstimator(15.8, 14.2);

		return result;
	}
	std::vector<MhaRuntime> get_avx_mha_runtime()
	{
		std::vector<MhaRuntime> result(2);
		// 10x8 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 10, 8, 256 };
		result[0].softmax_qk_kernel = mha_qk_avx_10x8;
		result[0].mult_by_v_kernel = gemm_avx_10x8;
		result[0].q_packing = pack_avx_10xK;
		result[0].kv_packing = pack_avx_8xK;
		result[0].perf_estimator = PerfEstimator(31.6, 16.7);

		// 10x8 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 10, 8, 512 };
		result[1].softmax_qk_kernel = mha_qk_avx_10x8;
		result[1].mult_by_v_kernel = gemm_avx_10x8;
		result[1].q_packing = pack_avx_10xK;
		result[1].kv_packing = pack_avx_8xK;
		result[1].perf_estimator = PerfEstimator(31.4, 14.6);

		return result;
	}
	std::vector<MhaRuntime> get_avx2_fma_mha_runtime()
	{
		std::vector<MhaRuntime> result(2);

		// 12x8 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 12, 8, 1024 };
		result[0].softmax_qk_kernel = mha_qk_avx2_12x8;
		result[0].mult_by_v_kernel = gemm_avx2_12x8;
		result[0].q_packing = pack_avx2_12xK;
		result[0].kv_packing = pack_avx_8xK;
		result[0].perf_estimator = PerfEstimator(62.8, 23.4);

		// 12x8 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 12, 8, 1024 };
		result[1].softmax_qk_kernel = mha_qk_avx2_12x8;
		result[1].mult_by_v_kernel = gemm_avx2_12x8;
		result[1].q_packing = pack_avx2_12xK;
		result[1].kv_packing = pack_avx_8xK;
		result[1].perf_estimator = PerfEstimator(61.2, 25.5);

		return result;
	}
	std::vector<MhaRuntime> get_avx512f_mha_runtime()
	{
		std::vector<MhaRuntime> result(2);

		// 24x16 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 24, 16, 1024 };
		result[0].softmax_qk_kernel = mha_qk_avx512f_24x16;
		result[0].mult_by_v_kernel = gemm_avx512f_24x16;
		result[0].q_packing = pack_avx512f_24xK;
		result[0].kv_packing = pack_avx512f_16xK;
		result[0].perf_estimator = PerfEstimator(114.3, 42.8);

		// 24x16 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 24, 16, 1024 };
		result[1].softmax_qk_kernel = mha_qk_avx512f_24x16;
		result[1].mult_by_v_kernel = gemm_avx512f_24x16;
		result[1].q_packing = pack_avx512f_24xK;
		result[1].kv_packing = pack_avx512f_16xK;
		result[1].perf_estimator = PerfEstimator(114.3, 42.8);

		return result;
	}

	template<typename T>
	void join_vectors(std::vector<T> &dst, const std::vector<T> &src)
	{
		dst.insert(dst.end(), src.begin(), src.end());
	}

	const std::vector<MhaRuntime>& get_mha_runtime_table(mlContext_t context)
	{
		static const std::vector<MhaRuntime> runtime_table = [context]()
		{
			std::vector<MhaRuntime> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
//			if (simd >= cpu::SimdLevel::AVX512F)
//				join_vectors(result, get_avx512f_mha_runtime());
			if (simd >= cpu::SimdLevel::AVX2)
				join_vectors(result, get_avx2_fma_mha_runtime());
//			if (simd >= cpu::SimdLevel::AVX)
//				join_vectors(result, get_avx_mha_runtime());
//			if (simd >= cpu::SimdLevel::SSE2)
//				join_vectors(result, get_sse2_mha_runtime());
			return result;
		}();
		assert(runtime_table.size() > 0);
		return runtime_table;
	}

	template<typename T>
	T clamp(T x, T lower, T upper) noexcept
	{
		assert(lower <= upper);
		return std::max(lower, std::min(upper, x));
	}
	template<typename SrcT, typename DstT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	cpu::float16 convert(float x) noexcept
	{
		return cpu::convert_fp32_to_fp16(x);
	}
	template<>
	float convert(cpu::float16 x) noexcept
	{
		return cpu::convert_fp16_to_fp32(x);
	}
	struct Index2D
	{
			int x, y;
	};
	template<typename T>
	void pack_bias(Fragment &dst, const BatchedMatrix &src, int head, int height, int width, int range)
	{
		assert(dst.is_packed());
		assert(dst.is_fp32());
		assert(dst.rows() >= height * width);
		assert(dst.columns() >= height * width);
		if (range >= (height - 1) and range >= (width - 1) and src.dtype() == DTYPE_FLOAT32)
		{ // no clipping and no conversion required -> plain memcpy
			for (int h1 = 0; h1 < height; h1++)
				for (int w1 = 0; w1 < width; w1++)
				{
					float *dst_ptr = dst.data<float>() + (h1 * width + w1) * dst.stride();
					for (int h2 = 0; h2 < height; h2++)
					{
						memcpy(dst_ptr, src.pointer_at(head, range + h2 - h1, range + 0 - w1), sizeof(float) * width);
						dst_ptr += width;
					}
				}
		}
		else
		{ // clipping or conversion required -> element by element copy
			for (int h1 = 0; h1 < height; h1++)
				for (int w1 = 0; w1 < width; w1++)
					for (int h2 = 0; h2 < height; h2++)
						for (int w2 = 0; w2 < width; w2++)
						{
							const int offset_h = range + clamp(h2 - h1, -range, range);
							const int offset_w = range + clamp(w2 - w1, -range, range);
							dst.at<float>(h1 * width + w1, h2 * width + w2) = convert<T, float>(src.at<T>(head, offset_h, offset_w));
						}
		}
	}

	Fragment get_subfragment(Fragment &frag, Position2D pos, Size2D size) noexcept
	{
		assert(frag.is_packed());
		void *shifted_ptr = frag.data<uint8_t>() + size_of(frag.dtype()) * frag.offset_at(pos.row, pos.column);
		Fragment result(shifted_ptr, frag.dtype(), frag.stride());
		result.mark_as_packed_with_size(size);
		return result;
	}
	void edge_softmax_sum(Fragment &softmax_sum, const Fragment &edge_qk) noexcept
	{
		for (int m = 0; m < edge_qk.columns(); m++)
		{
			float tmp = softmax_sum.at<float>(m, 0);
			for (int n = 0; n < edge_qk.rows(); n++)
				tmp += edge_qk.at<float>(n, m);
			softmax_sum.at<float>(m, 0) = tmp;
		}
	}

}

namespace ml
{

	void MhaRuntime::setup(mlContext_t context)
	{
		create_fragments(context);
	}
	void MhaRuntime::run()
	{
		const float scale = 1.0f / std::sqrt(head_dim);
		const float beta = 0.0f;
		Fragment null_fragment;
		Fragment out_fragment;

		for (int h = 0; h < num_heads; h++)
		{
			pack_fragment_bias(h);
			for (int b = 0; b < batch_size; b++)
			{
				Matrix Q(matrix_QKV.pointer_at(b, 0, 0 * embedding + h * head_dim), matrix_QKV.dtype(), tokens, head_dim, 3 * embedding);
				Matrix K(matrix_QKV.pointer_at(b, 0, 1 * embedding + h * head_dim), matrix_QKV.dtype(), tokens, head_dim, 3 * embedding);
				Matrix V(matrix_QKV.pointer_at(b, 0, 2 * embedding + h * head_dim), matrix_QKV.dtype(), tokens, head_dim, 3 * embedding);
				Matrix out(matrix_output.pointer_at(b, 0, h * head_dim), matrix_output.dtype(), tokens, head_dim, embedding);

				K_fragments.reset();
				V_fragments.reset();

				// loop over all tokens in Q matrix
				for (int inner_m = 0; inner_m < tokens; inner_m += inner_tile.M)
				{
					const int m_left = std::min(inner_tile.M, tokens - inner_m);

					pack_fragment_Q(Q, Q_fragment, inner_m, 0);
					softmax_sum_fragment.setall<float>(0.0f);
					temp_qk_fragment.mark_as_packed_with_size(Size2D(tokens, m_left));

					ArrayOfFragments::Iterator K_frag_iter = K_fragments.get_iterator();
					// loop over all tokens in K matrix
					for (int inner_n = 0; inner_n < tokens; inner_n += inner_tile.N)
					{
						const int n_left = std::min(inner_tile.N, tokens - inner_n);

						if (not K_frag_iter->is_packed())
							pack_fragment_K(K, *K_frag_iter, inner_n, 0);

						Fragment temp_qk_subfragment = get_subfragment(temp_qk_fragment, Position2D(inner_n, 0), Size2D(n_left, m_left));
						Fragment bias_subfragment = get_subfragment(bias_fragment, Position2D(inner_m, inner_n), Size2D(m_left, n_left));

						if (n_left == inner_tile.N)
							softmax_qk_kernel(temp_qk_subfragment, &scale, Q_fragment, *K_frag_iter, bias_subfragment, softmax_sum_fragment);
						else
						{
							softmax_qk_kernel(temp_qk_subfragment, &scale, Q_fragment, *K_frag_iter, bias_subfragment, null_fragment);
							edge_softmax_sum(softmax_sum_fragment, temp_qk_subfragment);
						}
						K_frag_iter.advance();
					}

					for (int i = 0; i < m_left; i++)
						softmax_sum_fragment.at<float>(i, 0) = 1.0f / softmax_sum_fragment.at<float>(i, 0);

					ArrayOfFragments::Iterator V_frag_iter = V_fragments.get_iterator();
					// loop over head dimension in V matrix
					for (int inner_n = 0; inner_n < head_dim; inner_n += inner_tile.N)
					{
						if (not V_frag_iter->is_packed())
							pack_fragment_V(V, *V_frag_iter, inner_n, 0);

						pack_fragment_out(out, out_fragment, inner_m, inner_n);
						mult_by_v_kernel(out_fragment, softmax_sum_fragment, temp_qk_fragment, *V_frag_iter, &beta, out_fragment, null_fragment,
								false);
						unpack_fragment_out(out, out_fragment, inner_m, inner_n);

						V_frag_iter.advance();
					}
				}
			}
		}
	}
	void MhaRuntime::create_fragments(mlContext_t context)
	{
		assert(matrix_QKV.data() != nullptr && matrix_output.data() != nullptr && matrix_bias.data() != nullptr); // all matrices have been set
		head_dim = embedding / num_heads;
		tokens = height * width;
		assert(inner_tile.M != 0 && inner_tile.N != 0); // tile sizes have been set
		const int num_K_fragments = divide_rounding_up(tokens, inner_tile.N);
		const int num_V_fragments = divide_rounding_up(head_dim, inner_tile.N);

		cpu::WorkspaceAllocator workspace_allocator(context);
		K_fragments = ArrayOfFragments(workspace_allocator, num_K_fragments);
		V_fragments = ArrayOfFragments(workspace_allocator, num_V_fragments);

		K_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(head_dim, inner_tile.N));
		V_fragments.create(workspace_allocator, type_configuration.compute_dtype, Size2D(tokens, inner_tile.N));

		Q_fragment = Fragment(workspace_allocator.get(size_of(DTYPE_FLOAT32) * inner_tile.M * head_dim, 4096), DTYPE_FLOAT32, inner_tile.M);

		const int qk_rows = round_to_multiple_of(tokens, inner_tile.N);
		temp_qk_fragment = Fragment(workspace_allocator.get(size_of(DTYPE_FLOAT32) * inner_tile.M * qk_rows, 4096), DTYPE_FLOAT32, inner_tile.M);

		const int bias_rows = round_to_multiple_of(tokens, inner_tile.M);
		const int bias_columns = round_to_multiple_of(tokens, 64 / sizeof(float));
		bias_fragment = Fragment(workspace_allocator.get(size_of(DTYPE_FLOAT32) * bias_rows * bias_columns, 4096), DTYPE_FLOAT32, bias_columns);
		bias_fragment.mark_as_packed_with_size(Size2D(tokens, tokens));
		bias_fragment.setall(0.0f);

		softmax_sum_fragment = Fragment(workspace_allocator.get(size_of(DTYPE_FLOAT32) * inner_tile.M, 64), DTYPE_FLOAT32, 1);
		softmax_sum_fragment.mark_as_packed_with_size(Size2D(inner_tile.M, 1));

		const Size2D tmp(inner_tile.M, inner_tile.N);
		const int fragment_size = size_of(matrix_output.dtype()) * tmp.rows * tmp.columns;
		edge_out_fragment = Fragment(workspace_allocator.get(fragment_size, 64), matrix_output.dtype(), matrix_output.stride());
	}
	void MhaRuntime::pack_fragment_Q(const Matrix &Q, Fragment &fragment, int m, int k)
	{
		const int k_to_pack = head_dim;
		const int m_to_pack = std::min(inner_tile.M, tokens - m);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, m_to_pack));

		const MatrixOp op = MatrixOp::TRANSPOSE;
		const Position2D pos = get_position(k, m, op);
		q_packing(fragment, Q, pos, op);
	}
	void MhaRuntime::pack_fragment_K(const Matrix &K, Fragment &fragment, int n, int k)
	{
		const int k_to_pack = head_dim;
		const int n_to_pack = std::min(inner_tile.N, tokens - n);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, n_to_pack));

		const MatrixOp op = MatrixOp::TRANSPOSE;
		const Position2D pos = get_position(k, n, op);
		kv_packing(fragment, K, pos, op);
	}
	void MhaRuntime::pack_fragment_V(const Matrix &V, Fragment &fragment, int n, int k)
	{
		const int k_to_pack = tokens;
		const int n_to_pack = std::min(inner_tile.N, head_dim - n);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, n_to_pack));

		const MatrixOp op = MatrixOp::NORMAL;
		const Position2D pos = get_position(k, n, op);
		kv_packing(fragment, V, pos, op);
	}
	void MhaRuntime::pack_fragment_out(Matrix &out, Fragment &fragment, int m, int n)
	{
		const int rows_to_pack = std::min(inner_tile.M, tokens - m);
		const int cols_to_pack = std::min(inner_tile.N, head_dim - n);

		const bool is_tile_full = (rows_to_pack == inner_tile.M) and (cols_to_pack == inner_tile.N);
		if (is_tile_full)
		{
			const Size2D tmp(inner_tile.M, inner_tile.N);
			fragment = Fragment(out.pointer_at(m, n), out.dtype(), out.stride());
			fragment.set_size(tmp, out.stride());
		}
		else
		{
			fragment = edge_out_fragment;
			fragment.set_size(Size2D(rows_to_pack, cols_to_pack), inner_tile.N);
			pack_def_MxK(fragment, out, Position2D(m, n), MatrixOp::NORMAL);
		}
	}
	void MhaRuntime::pack_fragment_bias(int head_idx)
	{
		if (matrix_bias.dtype() == DTYPE_FLOAT32)
			pack_bias<float>(bias_fragment, matrix_bias, head_idx, height, width, bias_range);
		else
			pack_bias<cpu::float16>(bias_fragment, matrix_bias, head_idx, height, width, bias_range);
	}
	void MhaRuntime::unpack_fragment_out(Matrix &out, Fragment &fragment, int m, int n)
	{
		const int rows_to_unpack = std::min(inner_tile.M, tokens - m);
		const int cols_to_unpack = std::min(inner_tile.N, head_dim - n);

		const bool is_tile_full = (rows_to_unpack == inner_tile.M) and (cols_to_unpack == inner_tile.N);
		if (not is_tile_full)
			unpack_def_MxK(out, Position2D(m, n), fragment);
	}

	MhaRuntime get_mha_runtime(mlContext_t context, mlDataType_t dtype, int tokens, int head_dim)
	{
		const MhaTypeConfiguration tc { dtype, dtype, dtype, dtype, DTYPE_FLOAT32 };

		const std::vector<MhaRuntime> &table = get_mha_runtime_table(context);
		MhaRuntime result;

		float max_gflops = std::numeric_limits<float>::lowest();
		for (auto iter = table.begin(); iter < table.end(); iter++)
		{
			if (iter->can_work_with_types(tc))
			{
				const float gflops = iter->get_expected_gflops(tokens, head_dim);
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
