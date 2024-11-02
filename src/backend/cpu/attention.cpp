/*
 * attention.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "gemm/Matrix.hpp"
#include "gemm/gemm_runtime.hpp"
#include "gemm/mha_runtime.hpp"
#include "indexers.hpp"
#include "fp16.hpp"

#include <cmath>

namespace
{
	using namespace ml;

	template<typename T>
	T clamp(T x, T lower, T upper) noexcept
	{
		assert(lower <= upper);
		return std::max(lower, std::min(upper, x));
	}
	int round_up(int x, int y) noexcept
	{
		const int tmp = x % y;
		return (tmp == 0) ? x : (x + y - tmp);
	}

	template<typename DstT, typename SrcT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	ml::cpu::float16 convert(float x) noexcept
	{
		return ml::cpu::convert_fp32_to_fp16(x);
	}
	template<>
	float convert(ml::cpu::float16 x) noexcept
	{
		return ml::cpu::convert_fp16_to_fp32(x);
	}

	float to_float(int32_t x) noexcept
	{
		return reinterpret_cast<const float*>(&x)[0];
	}
	float fast_exp(float x) noexcept
	{
		// maximum relative error = 0.628981%
		constexpr float a = (1 << 22) / float(M_LN2);
		constexpr int32_t b = 127 * (1 << 23) - 139160;
		const int32_t r = static_cast<int32_t>(a * x);
		const float s = to_float(b + r);
		const float t = to_float(b - r);
		return s / t;
	}

	struct PointerPack
	{
			std::vector<void*> q;
			std::vector<void*> k;
			std::vector<void*> v;
			std::vector<void*> qk;
			std::vector<void*> out;

			PointerPack(int num) :
					q(num),
					k(num),
					v(num),
					qk(num),
					out(num)
			{
			}
	};
	void* apply_offset(void *ptr, int offsetInBytes) noexcept
	{
		return reinterpret_cast<uint8_t*>(ptr) + offsetInBytes;
	}
	void calculate_pointers(const void *input, void *workspace, const void *output, PointerPack &pack, int batch_size, int tokens, int num_heads,
			int head_dim, int dtype_size, bool symmetric) noexcept
	{
		const Indexer<5> input_indexer(batch_size, tokens, 3 - symmetric, num_heads, head_dim);
		const Indexer<4> workspace_indexer(batch_size, num_heads, tokens, tokens);
		const Indexer<4> output_indexer(batch_size, tokens, num_heads, head_dim);

		int idx = 0;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++, idx++)
			{
				pack.q[idx] = apply_offset(const_cast<void*>(input), dtype_size * input_indexer.at(b, 0, 0, h, 0));
				if (symmetric)
					pack.k[idx] = pack.q[idx];
				else
					pack.k[idx] = apply_offset(const_cast<void*>(input), dtype_size * input_indexer.at(b, 0, 1, h, 0));
				pack.v[idx] = apply_offset(const_cast<void*>(input), dtype_size * input_indexer.at(b, 0, 2 - symmetric, h, 0));
				pack.qk[idx] = apply_offset(workspace, dtype_size * workspace_indexer.at(b, h, 0, 0));
				pack.out[idx] = apply_offset(const_cast<void*>(output), dtype_size * output_indexer.at(b, 0, h, 0));
			}
	}

	void gemm_batched(mlContext_t context, char opA, char opB, mlDataType_t dtype, int M, int N, int K, float alpha, std::vector<void*> &A, int lda,
			std::vector<void*> &B, int ldb, float beta, std::vector<void*> &C, int ldc, int batch_count)
	{
		GemmRuntime rt = get_gemm_runtime(context, dtype, opA, opB, M, N, K);

		for (int i = 0; i < batch_count; i++)
		{
			const Matrix matrix_aN(A[i], dtype, M, K, lda);
			const Matrix matrix_aT(A[i], dtype, K, M, lda);
			const Matrix matrix_bN(B[i], dtype, K, N, ldb);
			const Matrix matrix_bT(B[i], dtype, N, K, ldb);
			Matrix matrix_c(C[i], dtype, M, N, ldc);
			rt.setMatrixA((opA == 'n') ? matrix_aN : matrix_aT, opA);
			rt.setMatrixB((opB == 'n') ? matrix_bN : matrix_bT, opB);
			rt.setMatrixC(matrix_c);
			rt.setMatrixD(matrix_c);
			if (i == 0)
			{
				rt.setScalingFactors(alpha, beta);
				rt.setup(context);
			}
			rt.run();
		}
	}
	template<typename T, bool UseBias>
	void softmax_forward_in_place(void *input, const void *weights, int batch_size, int num_heads, int height, int width, int weights_size,
			void *workspace)
	{
		const T *weights_ptr = ml::getPointer<T>(weights);
		float *input_cache_ptr = ml::getPointer<float>(workspace);
		float *weights_cache_ptr = input_cache_ptr + height * width;

		const int range = (weights_size - 1) / 2;

		Indexer<2> weight_indexer(weights_size, round_up(weights_size, 4));

		for (int h = 0; h < num_heads; h++)
		{
			if (UseBias)
			{
				for (int i = 0; i < weights_size * round_up(weights_size, 4); i++)
					weights_cache_ptr[i] = convert<float>(weights_ptr[i]);
			}

			for (int b = 0; b < batch_size; b++)
			{
				Indexer<4> input_indexer(batch_size, num_heads, height * width, height * width);
				T *input_ptr = ml::getPointer<T>(input) + input_indexer.at(b, h, 0, 0);
				for (int h1 = 0; h1 < height; h1++)
					for (int w1 = 0; w1 < width; w1++)
					{
						float max_value = -1.0e+32f;
						int idx = 0;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								float bias = 0.0f;
								if (UseBias)
								{
									const int offset_h = range + clamp(h2 - h1, -range, range);
									const int offset_w = range + clamp(w2 - w1, -range, range);
									bias = weights_cache_ptr[weight_indexer.at(offset_h, offset_w)];
								}
								const float tmp = convert<float>(input_ptr[idx]) + bias;
								max_value = std::max(max_value, tmp);
								input_cache_ptr[idx] = tmp;
								idx++;
							}

						float sum = 0.0f;
						for (int i = 0; i < height * width; i++)
						{
							const T tmp = std::exp(input_cache_ptr[i] - max_value);
							sum += tmp;
							input_cache_ptr[i] = tmp;
						}

						const float scale = 1.0f / sum;
						for (int i = 0; i < height * width; i++)
							input_ptr[i] = convert<T>(input_cache_ptr[i] * scale);

						input_ptr += height * width;
					}
			}

			weights_ptr += weights_size * round_up(weights_size, 4);
		}
	}
	template<bool UseBias>
	void softmax_backward_in_place(void *gradient, void *weights_update, const void *output, int batch_size, int num_heads, int height, int width,
			int weights_size)
	{
		const float *output_ptr = ml::getPointer<float>(output);
		float *gradient_ptr = ml::getPointer<float>(gradient);
		float *weights_update_ptr = ml::getPointer<float>(weights_update);

		const int range = (weights_size - 1) / 2;

		Indexer<3> weight_indexer(num_heads, weights_size, round_up(weights_size, 4));
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++)
			{
				for (int h1 = 0; h1 < height; h1++)
					for (int w1 = 0; w1 < width; w1++)
					{
						float tmp = 0;
						for (int i = 0; i < height * width; i++)
							tmp += gradient_ptr[i] * output_ptr[i];

						int idx = 0;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const float dx = output_ptr[idx] * (gradient_ptr[idx] - tmp);
								gradient_ptr[idx] = dx;
								if (UseBias)
								{
									const int offset_h = range + clamp(h2 - h1, -range, range);
									const int offset_w = range + clamp(w2 - w1, -range, range);
									weights_update_ptr[weight_indexer.at(h, offset_h, offset_w)] += dx;
								}
								idx++;
							}
						output_ptr += height * width;
						gradient_ptr += height * width;
					}
			}
	}

	void fused_mha_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlDataType_t dtype, const void *input, void *output,
			const void *weights, void *workspace, void *backward_data, int num_heads, bool symmetric)
	{
		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / (3 - symmetric);
		const int head_dim = embedding / num_heads;

		MhaRuntime rt = get_mha_runtime(context, dtype, tokens, head_dim);
		rt.setInput(input, input_shape, dtype, num_heads, symmetric);
		rt.setOutput(output, make_shape( { batch_size, height, width, embedding }), dtype);
		rt.setBias(weights, weights_shape, dtype);
		rt.setup(context);
		rt.run();
	}
}

namespace ml
{
	int cpu_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, int num_heads, bool training)
	{
		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int tokens = input_shape.dim[1] * input_shape.dim[2];

		int result = batch_size * num_heads * tokens * tokens;
		if (training)
			result *= 2;
		return result;
	}

	void cpu_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlShape_t bias_shape,
			mlDataType_t dtype, const void *input, void *output, const void *weights, const void *bias, const void *mask, void *workspace,
			void *backward_data, int num_heads, bool symmetric)
	{
		if (backward_data == nullptr)
		{
			fused_mha_forward(context, input_shape, weights_shape, dtype, input, output, weights, workspace, backward_data, num_heads, symmetric);
			return;
		}

		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / (3 - symmetric);
		const int head_dim = embedding / num_heads;
		const bool use_bias = weights != nullptr;
		const int range = use_bias ? weights_shape.dim[1] : 0;

		const int qkv_stride = input_shape.dim[3];

		const int num_pointers = batch_size * num_heads;
		void *local_workspace = ml::cpu::Context::getWorkspace(context);

		PointerPack pack(num_pointers);

		void *qk_tensor_ptr = (backward_data == nullptr) ? workspace : backward_data;
		calculate_pointers(input, qk_tensor_ptr, output, pack, batch_size, tokens, num_heads, head_dim, size_of(dtype), symmetric);

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', dtype, tokens, tokens, head_dim, scale, pack.q, qkv_stride, pack.k, qkv_stride, 0.0f, pack.qk, tokens,
				num_pointers);

		if (use_bias)
			softmax_forward_in_place<float, true>(qk_tensor_ptr, weights, batch_size, num_heads, height, width, range, local_workspace);
		else
			softmax_forward_in_place<float, false>(qk_tensor_ptr, nullptr, batch_size, num_heads, height, width, 0, local_workspace);

		gemm_batched(context, 'n', 'n', dtype, tokens, head_dim, tokens, 1.0f, pack.qk, tokens, pack.v, qkv_stride, 0.0f, pack.out, embedding,
				num_pointers);
	}
	void cpu_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlShape_t bias_shape,
			const void *input, const void *weights, const void *bias, const void *mask, void *gradient_prev, void *gradient_next,
			void *weights_update, void *bias_update, void *workspace, void *backward_data, int num_heads, bool symmetric)
	{
		assert(input_shape.rank == 4);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int tokens = height * width;
		const int embedding = input_shape.dim[3] / (3 - symmetric);
		const int head_dim = embedding / num_heads;
		const bool use_bias = weights != nullptr;
		const int range = use_bias ? weights_shape.dim[1] : 0;
		const int qkv_stride = input_shape.dim[3];

		const int offset = batch_size * num_heads * tokens * tokens * size_of(DTYPE_FLOAT32);
		void *qk_tensor_ptr = (backward_data == nullptr) ? workspace : backward_data;
		void *dqk_tensor_ptr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace) + offset);

		const int num_pointers = batch_size * num_heads;
		void *local_workspace = ml::cpu::Context::getWorkspace(context);

		PointerPack forward_pack(num_pointers);
		PointerPack backward_pack(num_pointers);

		const float scale = 1.0f / std::sqrt(head_dim);
		calculate_pointers(input, qk_tensor_ptr, nullptr, forward_pack, batch_size, tokens, num_heads, head_dim, size_of(DTYPE_FLOAT32), symmetric);
		calculate_pointers(gradient_prev, dqk_tensor_ptr, gradient_next, backward_pack, batch_size, tokens, num_heads, head_dim,
				size_of(DTYPE_FLOAT32), symmetric);

		if (backward_data == nullptr)
		{
			gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, scale, forward_pack.q, qkv_stride, forward_pack.k, qkv_stride,
					0.0f, forward_pack.qk, tokens, num_pointers);
			if (use_bias)
				softmax_forward_in_place<float, true>(qk_tensor_ptr, weights, batch_size, num_heads, height, width, range, local_workspace);
			else
				softmax_forward_in_place<float, false>(qk_tensor_ptr, nullptr, batch_size, num_heads, height, width, 0, local_workspace);
		}

		// dqk = dy * V^T
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, 1.0f, backward_pack.out, embedding, forward_pack.v, qkv_stride, 0.0f,
				backward_pack.qk, tokens, num_pointers);
		// dV = qk^T * dy
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, forward_pack.qk, tokens, backward_pack.out, embedding, 0.0f,
				backward_pack.v, qkv_stride, num_pointers);

		if (use_bias)
		{
			assert(weights_shape.rank == 3);
			softmax_backward_in_place<true>(dqk_tensor_ptr, weights_update, qk_tensor_ptr, batch_size, num_heads, height, width, range);
		}
		else
			softmax_backward_in_place<false>(dqk_tensor_ptr, nullptr, qk_tensor_ptr, batch_size, num_heads, height, width, 0);

		// dQ = dqk * K
		gemm_batched(context, 'n', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, scale, backward_pack.qk, tokens, forward_pack.k, qkv_stride, 0.0f,
				backward_pack.q, qkv_stride, num_pointers);

		const float beta = symmetric ? 1.0f : 0.0f;
		// dK = dqk^T * Q
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, scale, backward_pack.qk, tokens, forward_pack.q, qkv_stride, beta,
				backward_pack.k, qkv_stride, num_pointers);
	}
} /* namespace ml */
