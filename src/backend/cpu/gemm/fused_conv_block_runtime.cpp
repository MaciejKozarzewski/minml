/*
 * fused_conv_block_runtime.cpp
 *
 *  Created on: Apr 16, 2025
 *      Author: Maciej Kozarzewski
 */

#include "fused_conv_block_runtime.hpp"

#include "../utils.hpp"
#include "../fp16.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"
#include "gemm_kernels.hpp"
#include "gemm_runtime.hpp"

#include <limits>
#include <cstring>
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

	std::vector<FCBRuntime> get_sse2_fcb_runtime()
	{
		std::vector<FCBRuntime> result(1);

		// 4x8 fp32
		result[0].dtype = DTYPE_FLOAT32;
		result[0].inner_tile = { 4, 8, 1024 };
		result[0].dwconv_kernel = depthwise_conv_sse2_4x8;
		result[0].first_conv_kernel = fused_conv_block_stage_1_sse2_4x8;
		result[0].second_conv_kernel = gemm_sse2_4x8;
		result[0].input_packing = pack_sse2_4xK;
		result[0].weights_packing = pack_sse2_8xK;
		result[0].perf_estimator = PerfEstimator(15.8, 14.2);

		return result;
	}
	std::vector<FCBRuntime> get_avx_fcb_runtime()
	{
		std::vector<FCBRuntime> result(2);

		// 10x8 fp32
		result[0].dtype = DTYPE_FLOAT32;
		result[0].inner_tile = { 10, 8, 1024 };
		result[0].dwconv_kernel = depthwise_conv_avx_10x8;
		result[0].first_conv_kernel = fused_conv_block_stage_1_avx_10x8;
		result[0].second_conv_kernel = gemm_avx_10x8;
		result[0].input_packing = pack_avx_10xK;
		result[0].weights_packing = pack_avx_8xK;
		result[0].perf_estimator = PerfEstimator(31.6, 16.7);

		// 10x8 fp16/fp32
		result[1].dtype = DTYPE_FLOAT16;
		result[1].inner_tile = { 10, 8, 1024 };
		result[1].dwconv_kernel = depthwise_conv_avx_10x8;
		result[1].first_conv_kernel = fused_conv_block_stage_1_avx_10x8;
		result[1].second_conv_kernel = gemm_avx_10x8;
		result[1].input_packing = pack_avx_10xK;
		result[1].weights_packing = pack_avx_8xK;
		result[1].perf_estimator = PerfEstimator(31.4, 14.6);

		return result;
	}
	std::vector<FCBRuntime> get_avx2_fma_fcb_runtime()
	{
		std::vector<FCBRuntime> result(2);

		// 12x8 fp32
		result[0].dtype = DTYPE_FLOAT32;
		result[0].inner_tile = { 12, 8, 1024 };
		result[0].dwconv_kernel = depthwise_conv_avx2_12x8;
		result[0].first_conv_kernel = fused_conv_block_stage_1_avx2_12x8;
		result[0].second_conv_kernel = gemm_avx2_12x8;
		result[0].input_packing = pack_avx2_12xK;
		result[0].weights_packing = pack_avx_8xK;
		result[0].perf_estimator = PerfEstimator(62.8, 23.4);

		// 12x8 fp16/fp32
		result[1].dtype = DTYPE_FLOAT16;
		result[1].inner_tile = { 12, 8, 1024 };
		result[1].dwconv_kernel = depthwise_conv_avx2_12x8;
		result[1].first_conv_kernel = fused_conv_block_stage_1_avx2_12x8;
		result[1].second_conv_kernel = gemm_avx2_12x8;
		result[1].input_packing = pack_avx2_12xK;
		result[1].weights_packing = pack_avx_8xK;
		result[1].perf_estimator = PerfEstimator(61.2, 25.5);

		return result;
	}
	std::vector<FCBRuntime> get_avx512f_fcb_runtime()
	{
		std::vector<FCBRuntime> result(2);

		// 24x16 fp32
		result[0].dtype = DTYPE_FLOAT32;
		result[0].inner_tile = { 24, 16, 1024 };
		result[0].dwconv_kernel = depthwise_conv_avx512f_24x16;
		result[0].first_conv_kernel = fused_conv_block_stage_1_avx512f_24x16;
		result[0].second_conv_kernel = gemm_avx512f_24x16;
		result[0].input_packing = pack_avx512f_24xK;
		result[0].weights_packing = pack_avx512f_16xK;
		result[0].perf_estimator = PerfEstimator(114.3, 42.8);

		// 24x16 fp16/fp32
		result[1].dtype = DTYPE_FLOAT16;
		result[1].inner_tile = { 24, 16, 1024 };
		result[1].dwconv_kernel = depthwise_conv_avx512f_24x16;
		result[1].first_conv_kernel = fused_conv_block_stage_1_avx512f_24x16;
		result[1].second_conv_kernel = gemm_avx512f_24x16;
		result[1].input_packing = pack_avx512f_24xK;
		result[1].weights_packing = pack_avx512f_16xK;
		result[1].perf_estimator = PerfEstimator(114.3, 42.8);

		return result;
	}

	template<typename T>
	void join_vectors(std::vector<T> &dst, const std::vector<T> &src)
	{
		dst.insert(dst.end(), src.begin(), src.end());
	}

	const std::vector<FCBRuntime>& get_fcb_runtime_table(mlContext_t context)
	{
		static const std::vector<FCBRuntime> runtime_table = [context]()
		{
			std::vector<FCBRuntime> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
			if (simd >= cpu::SimdLevel::AVX512F)
				join_vectors(result, get_avx512f_fcb_runtime());
			if (simd >= cpu::SimdLevel::AVX2)
				join_vectors(result, get_avx2_fma_fcb_runtime());
			if (simd >= cpu::SimdLevel::AVX)
				join_vectors(result, get_avx_fcb_runtime());
			if (simd >= cpu::SimdLevel::SSE2)
				join_vectors(result, get_sse2_fcb_runtime());
			return result;
		}();
		assert(runtime_table.size() > 0);
		return runtime_table;
	}

	Fragment get_subfragment(Fragment &frag, Position2D pos, Size2D size) noexcept
	{
		if (frag.is_packed())
		{
			void *shifted_ptr = frag.data<uint8_t>() + size_of(frag.dtype()) * frag.offset_at(pos.row, pos.column);
			Fragment result(shifted_ptr, frag.dtype(), frag.stride());
			result.mark_as_packed_with_size(size);
			return result;
		}
		else
			return Fragment();
	}

	void fill_indices_table(int *indices, int batch_size, int height, int width) noexcept
	{
		int idx = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h++)
				for (int w = 0; w < width; w++)
				{
					indices[idx + 0] = b;
					indices[idx + 1] = h;
					indices[idx + 2] = w;
					idx += 3;
				}
	}
	double sum_fragment(const Fragment &fragment)
	{
		assert(fragment.is_fp32());
		double result = 0.0;
		for (int i = 0; i < fragment.rows(); i++)
			for (int j = 0; j < fragment.columns(); j++)
				result += fragment.at<float>(i, j);
		return result;
	}

	void find_scales(float *scales, const Matrix &weights) noexcept
	{
		const float *w_ptr = reinterpret_cast<const float*>(weights.data());
		for (int j = 0; j < weights.columns(); j++)
			scales[j] = w_ptr[j];
		w_ptr += weights.columns();
		for (int i = 1; i < weights.rows(); i++)
		{
			for (int j = 0; j < weights.columns(); j++)
				scales[j] = std::max(scales[j], std::abs(w_ptr[j]));
			w_ptr += weights.columns();
		}
		for (int j = 0; j < weights.columns(); j++)
			scales[j] = 4095.0f / scales[j];
	}
	void find_scales(float *scales, const Fragment &fragment) noexcept
	{
		const float *w_ptr = reinterpret_cast<const float*>(fragment.data());
		for (int j = 0; j < fragment.columns(); j++)
			scales[j] = w_ptr[j];
		w_ptr += fragment.columns();
		for (int i = 1; i < fragment.rows(); i++)
		{
			for (int j = 0; j < fragment.columns(); j++)
				scales[j] = std::max(scales[j], std::abs(w_ptr[j]));
			w_ptr += fragment.columns();
		}
		for (int j = 0; j < fragment.columns(); j++)
			scales[j] = 4095.0f / scales[j];
	}
}

namespace ml
{

	void FCBRuntime::setup(mlContext_t context)
	{
		create_fragments(context);
	}
	void FCBRuntime::run()
	{
		int args[6] = { batch_size, height, width, channels, dw_kernel_size, dw_kernel_size };
		fill_indices_table(indices, batch_size, height, width);

		const float alpha = 1.0f;
		const float beta = 1.0f;
		Fragment alpha_fragment(&alpha, DTYPE_FLOAT32, 0);
		alpha_fragment.mark_as_packed_with_size( { 1, 1 });

		Fragment in_fragment;
		Fragment out_fragment;

		for (int inner_m = 0; inner_m < batch_size * height * width; inner_m += inner_tile.M)
		{
			const int m_left = std::min(inner_tile.M, batch_size * height * width - inner_m);

			Matrix tmp_out(dw_out_matrix.data(), dw_out_matrix.dtype(), m_left, dw_out_matrix.columns(), dw_out_matrix.stride());
			dwconv_kernel(tmp_out, input_matrix, dwconv_weights, dwconv_bias, args, indices + 3 * inner_m);
			pack_fragment_input(dwconv_output_fragment, inner_m);
//			quantize_avx2_12xK(quantized_fragment, dwconv_output_fragment);

			first_conv_output_fragment.mark_as_packed_with_size(Size2D(hidden_dim, std::min(inner_tile.M, m_left)));

			ArrayOfFragments::Iterator w1_frag_iter = w1_fragments.get_iterator();
			ArrayOfFragments::Iterator b1_frag_iter = b1_fragments.get_iterator();
			for (int inner_n = 0; inner_n < hidden_dim; inner_n += inner_tile.N)
			{
				const int n_left = std::min(inner_tile.N, hidden_dim - inner_n);

				if (not w1_frag_iter->is_packed())
					pack_fragment_weights(first_conv_weights, *w1_frag_iter, inner_n);
				if (not b1_frag_iter->is_packed())
					pack_fragment_bias(first_conv_bias, *b1_frag_iter, inner_n);

				Fragment temp_subfragment = get_subfragment(first_conv_output_fragment, Position2D(inner_n, 0), Size2D(n_left, m_left));

				first_conv_kernel(temp_subfragment, dwconv_output_fragment, *w1_frag_iter, *b1_frag_iter);

				w1_frag_iter.advance();
				b1_frag_iter.advance();
			}
//			quantize_avx2_12xK(quantized_fragment, first_conv_output_fragment);

			ArrayOfFragments::Iterator w2_frag_iter = w2_fragments.get_iterator();
			ArrayOfFragments::Iterator b2_frag_iter = b2_fragments.get_iterator();
			for (int inner_n = 0; inner_n < channels; inner_n += inner_tile.N)
			{
				if (not w2_frag_iter->is_packed())
					pack_fragment_weights(second_conv_weights, *w2_frag_iter, inner_n);
				if (not b2_frag_iter->is_packed())
					pack_fragment_bias(second_conv_bias, *b2_frag_iter, inner_n);

				pack_fragment_out(output_matrix, out_fragment, inner_m, inner_n);
				pack_fragment_out(input_matrix, in_fragment, inner_m, inner_n);

				second_conv_kernel(out_fragment, alpha_fragment, first_conv_output_fragment, *w2_frag_iter, &beta, in_fragment, *b2_frag_iter, false);

				unpack_fragment_out(out_fragment, inner_m, inner_n);

				w2_frag_iter.advance();
				b2_frag_iter.advance();
			}
		}
	}
	void FCBRuntime::create_fragments(mlContext_t context)
	{
		assert(inner_tile.M != 0 && inner_tile.N != 0); // tile sizes have been set
		const int num_1_fragments = divide_rounding_up(hidden_dim, inner_tile.N);
		const int num_2_fragments = divide_rounding_up(channels, inner_tile.N);

		cpu::WorkspaceAllocator workspace_allocator(context);
		w1_fragments = ArrayOfFragments(workspace_allocator, num_1_fragments);
		w2_fragments = ArrayOfFragments(workspace_allocator, num_2_fragments);
		w1_fragments.create(workspace_allocator, DTYPE_FLOAT32, Size2D(channels, inner_tile.N), 4096);
		w2_fragments.create(workspace_allocator, DTYPE_FLOAT32, Size2D(hidden_dim, inner_tile.N), 4096);

		b1_fragments = ArrayOfFragments(workspace_allocator, num_1_fragments);
		b2_fragments = ArrayOfFragments(workspace_allocator, num_2_fragments);
		b1_fragments.create(workspace_allocator, DTYPE_FLOAT32, Size2D(1, inner_tile.N), 64);
		b2_fragments.create(workspace_allocator, DTYPE_FLOAT32, Size2D(1, inner_tile.N), 64);

		const int dw_out_size = inner_tile.M * channels;
		dw_out_matrix = Matrix(workspace_allocator.get(dw_out_size * size_of(DTYPE_FLOAT32), 4096), DTYPE_FLOAT32, inner_tile.M, channels, channels);

		const int dwconv_rows = round_to_multiple_of(channels, inner_tile.N);
		dwconv_output_fragment = Fragment(workspace_allocator.get(size_of(DTYPE_FLOAT32) * inner_tile.M * dwconv_rows, 4096), DTYPE_FLOAT32,
				inner_tile.M);

		const int first_conv_rows = round_to_multiple_of(hidden_dim, inner_tile.N);
		first_conv_output_fragment = Fragment(workspace_allocator.get(size_of(DTYPE_FLOAT32) * inner_tile.M * first_conv_rows, 4096), DTYPE_FLOAT32,
				inner_tile.M);

		const Size2D tmp(inner_tile.M, inner_tile.N);
		const int fragment_size = size_of(output_matrix.dtype()) * tmp.rows * tmp.columns;
		edge_output_fragment = Fragment(workspace_allocator.get(fragment_size, 64), output_matrix.dtype(), output_matrix.stride());

		quantized_fragment = Fragment(workspace_allocator.get(dw_out_size * size_of(DTYPE_FLOAT32), 4096), DTYPE_FLOAT32, inner_tile.M);

		indices = reinterpret_cast<int*>(workspace_allocator.get(3 * sizeof(int) * batch_size * height * width, 4096));
	}
	void FCBRuntime::pack_fragment_input(Fragment &fragment, int m)
	{
		const int k_to_pack = channels;
		const int m_to_pack = std::min(inner_tile.M, batch_size * height * width - m);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, m_to_pack));

		input_packing(fragment, dw_out_matrix, Position2D(0, 0), MatrixOp::TRANSPOSE);
	}
	void FCBRuntime::pack_fragment_weights(const Matrix &weights, Fragment &fragment, int n)
	{
		const int k_to_pack = weights.columns();
		const int n_to_pack = std::min(inner_tile.N, weights.rows() - n);
		fragment.mark_as_packed_with_size(Size2D(k_to_pack, n_to_pack));

		const MatrixOp op = MatrixOp::TRANSPOSE;
		const Position2D pos = get_position(0, n, op);
		weights_packing(fragment, weights, pos, op);
//		quantize_avx2_8xK(quantized_fragment, fragment);
	}
	void FCBRuntime::pack_fragment_out(Matrix &out, Fragment &fragment, int m, int n)
	{
		const int rows_to_pack = std::min(inner_tile.M, batch_size * height * width - m);
		const int cols_to_pack = std::min(inner_tile.N, channels - n);

		const bool is_tile_full = (rows_to_pack == inner_tile.M) and (cols_to_pack == inner_tile.N);
		if (is_tile_full)
		{
			const Size2D tmp(inner_tile.M, inner_tile.N);
			fragment = Fragment(out.pointer_at(m, n), out.dtype(), out.stride());
			fragment.set_size(tmp, out.stride());
		}
		else
		{
			fragment = edge_output_fragment;
			fragment.set_size(Size2D(rows_to_pack, cols_to_pack), inner_tile.N);
			pack_def_MxK(fragment, out, Position2D(m, n), MatrixOp::NORMAL);
		}
	}
	void FCBRuntime::pack_fragment_bias(const Matrix &bias, Fragment &fragment, int n)
	{
		const int n_to_pack = std::min(inner_tile.N, bias.columns() - n);
		fragment.mark_as_packed_with_size(Size2D(1, n_to_pack));

		const Position2D pos = get_position(0, n, MatrixOp::NORMAL);
		weights_packing(fragment, bias, pos, MatrixOp::NORMAL);
	}
	void FCBRuntime::unpack_fragment_out(Fragment &fragment, int m, int n)
	{
		const int rows_to_unpack = std::min(inner_tile.M, batch_size * height * width - m);
		const int cols_to_unpack = std::min(inner_tile.N, channels - n);

		const bool is_tile_full = (rows_to_unpack == inner_tile.M) and (cols_to_unpack == inner_tile.N);
		if (not is_tile_full)
			unpack_def_MxK(output_matrix, Position2D(m, n), fragment);
	}

	FCBRuntime get_fcb_runtime(mlContext_t context, mlDataType_t dtype)
	{
		const std::vector<FCBRuntime> &table = get_fcb_runtime_table(context);
		FCBRuntime result;

		float max_gflops = std::numeric_limits<float>::lowest();
		for (auto iter = table.begin(); iter < table.end(); iter++)
		{
			if (iter->can_work_with_types(dtype))
			{
				const float gflops = 1.0f; //iter->get_expected_gflops(tokens, head_dim);
				if (gflops > max_gflops)
				{
					result = *iter;
					max_gflops = gflops;
				}
			}
		}
		return result;
	}

	void cpu_fused_conv_block_forward(mlContext_t context, const mlTensor_t input, const mlTensor_t dwconv_weights, const mlTensor_t dwconv_bias,
			const mlTensor_t first_conv_weights, const mlTensor_t first_conv_bias, const mlTensor_t second_conv_weights,
			const mlTensor_t second_conv_bias, mlTensor_t output)
	{
		FCBRuntime rt = get_fcb_runtime(context, input.dtype);
		rt.setInput(input);
		rt.setDepthwiseConv(dwconv_weights, dwconv_bias);
		rt.setFirstConv(first_conv_weights, first_conv_bias);
		rt.setSecondConv(second_conv_weights, second_conv_bias);
		rt.setOutput(output);
		rt.setup(context);
		rt.run();
	}

	void cpu_depthwise_conv_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y)
	{
		const int first_dim = volume_without_last_dim(x);
		const int last_dim = get_last_dim(x);
		assert(w.dim[0] == w.dim[1]); // square kernel

		cpu::WorkspaceAllocator workspace_allocator(context);

		void *zero_line = workspace_allocator.get(sizeof(float) * last_dim, 4096);
		std::memset(zero_line, 0, sizeof(float) * last_dim);

		Matrix input_matrix(x.data, x.dtype, 1, last_dim, last_dim);
		Matrix weight_matrix(w.data, w.dtype, w.dim[0] * w.dim[1], w.dim[3], w.dim[3]);
		Matrix bias_matrix = (b.data == nullptr) ? Matrix(zero_line, b.dtype, 1, last_dim, last_dim) : Matrix(b.data, b.dtype, 1, b.dim[0], b.dim[0]);

		int args[6] = { x.dim[0], x.dim[1], x.dim[2], x.dim[3], w.dim[0], w.dim[1] };

		int *indices = reinterpret_cast<int*>(workspace_allocator.get(first_dim * 3, 4096));
		fill_indices_table(indices, x.dim[0], x.dim[1], x.dim[2]);

		Matrix output_matrix(y.data, y.dtype, first_dim, last_dim, last_dim);
		const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
		if (simd >= cpu::SimdLevel::AVX512F)
		{
			depthwise_conv_avx512f_24x16(output_matrix, input_matrix, weight_matrix, bias_matrix, args, indices);
			return;
		}
		if (simd >= cpu::SimdLevel::AVX2)
		{
			depthwise_conv_avx2_12x8(output_matrix, input_matrix, weight_matrix, bias_matrix, args, indices);
			return;
		}
		if (simd >= cpu::SimdLevel::AVX)
		{
			depthwise_conv_avx_10x8(output_matrix, input_matrix, weight_matrix, bias_matrix, args, indices);
			return;
		}
		if (simd >= cpu::SimdLevel::SSE2)
		{
			depthwise_conv_sse2_4x8(output_matrix, input_matrix, weight_matrix, bias_matrix, args, indices);
			return;
		}

	}

} /* namespace ml */

