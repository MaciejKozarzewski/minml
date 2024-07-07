/*
 * test_attention.cpp
 *
 *  Created on: Jun 14, 2024
 *      Author: Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/utils/testing_util.hpp>

#include <cmath>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	template<typename T>
	T clamp(T x, T lower, T upper) noexcept
	{
		assert(lower <= upper);
		return std::max(lower, std::min(upper, x));
	}

	Tensor baseline_extract_qkv(char c, const Tensor &qkv)
	{
		assert(qkv.rank() == 4);
		assert(qkv.lastDim() % 3 == 0);
		const int B = qkv.firstDim();
		const int T = qkv.dim(1) * qkv.dim(2);
		const int C = qkv.lastDim() / 3;
		const Tensor tmp = qkv.view( { B, T, 3, C });
		Tensor result( { B, T, C }, qkv.dtype(), qkv.device());

		const int index = (c == 'Q') ? 0 : ((c == 'K') ? 1 : 2);

		for (int b = 0; b < B; b++)
			for (int t = 0; t < T; t++)
				for (int c = 0; c < C; c++)
					result.set(tmp.get( { b, t, index, c }), { b, t, c });
		return result;
	}
	void baseline_insert_qkv(char c, const Tensor &x, Tensor &qkv)
	{
		assert(qkv.rank() == 4);
		assert(qkv.lastDim() % 3 == 0);
		const int B = qkv.firstDim();
		const int T = qkv.dim(1) * qkv.dim(2);
		const int C = qkv.lastDim() / 3;
		Tensor dst = qkv.view( { B, T, 3, C });

		const int index = (c == 'Q') ? 0 : ((c == 'K') ? 1 : 2);

		Tensor tmp = x.view( { B, T, C });
		for (int b = 0; b < B; b++)
			for (int t = 0; t < T; t++)
				for (int c = 0; c < C; c++)
					dst.set(tmp.get( { b, t, c }), { b, t, index, c });
	}
	Tensor split_heads(const Tensor &x, int num_heads)
	{
		assert(x.rank() == 3);
		assert(x.dim(2) % num_heads == 0);
		const int B = x.dim(0);
		const int T = x.dim(1);
		const int H = num_heads;
		const int C = x.dim(2) / num_heads;
		const Tensor tmp = x.view( { B, T, H, C });
		Tensor result( { B, H, T, C }, x.dtype(), x.device());

		for (int b = 0; b < B; b++)
			for (int t = 0; t < T; t++)
				for (int h = 0; h < H; h++)
					for (int c = 0; c < C; c++)
						result.set(tmp.get( { b, t, h, c }), { b, h, t, c });
		return result;
	}
	Tensor combine_heads(const Tensor &x, const Shape &input_shape)
	{
		assert(x.rank() == 4);
		const int B = x.dim(0);
		const int H = x.dim(1);
		const int T = x.dim(2);
		const int C = x.dim(3);
		Tensor result( { B, T, H, C }, x.dtype(), x.device());

		for (int b = 0; b < B; b++)
			for (int t = 0; t < T; t++)
				for (int h = 0; h < H; h++)
					for (int c = 0; c < C; c++)
						result.set(x.get( { b, h, t, c }), { b, t, h, c });
		result.reshape( { B, input_shape.dim(1), input_shape.dim(2), H * C });
		return result;
	}
	Tensor baseline_batched_gemm(const Tensor &A, const Tensor &B, float alpha, char opA, char opB)
	{
		assert(A.rank() == 4);
		assert(B.rank() == 4);
		const int M = (opA == 'n') ? A.dim(2) : A.dim(3);
		const int N = (opB == 'n') ? B.dim(3) : B.dim(2);
		const int K = (opA == 'n') ? A.dim(3) : A.dim(2);

		Tensor result( { A.dim(0), A.dim(1), M, N }, A.dtype(), A.device());
		for (int b = 0; b < A.dim(0); b++)
			for (int h = 0; h < A.dim(1); h++)
				for (int m = 0; m < M; m++)
					for (int n = 0; n < N; n++)
					{
						float tmp = 0.0f;
						if (opA == 'n')
						{
							if (opB == 'n')
								for (int k = 0; k < K; k++)
									tmp += A.get( { b, h, m, k }) * B.get( { b, h, k, n });
							else
								for (int k = 0; k < K; k++)
									tmp += A.get( { b, h, m, k }) * B.get( { b, h, n, k });
						}
						else
						{
							if (opB == 'n')
								for (int k = 0; k < K; k++)
									tmp += A.get( { b, h, k, m }) * B.get( { b, h, k, n });
							else
								for (int k = 0; k < K; k++)
									tmp += A.get( { b, h, k, m }) * B.get( { b, h, n, k });
						}
						result.set(alpha * tmp, { b, h, m, n });
					}
		return result;
	}
	void baseline_softmax_forward(Tensor &qk, const Tensor &weights, const Shape &input_shape)
	{
		const int batch_size = input_shape[0];
		const int height = input_shape[1];
		const int width = input_shape[2];
		const int num_heads = weights.firstDim();
		assert(qk.dim(1) == num_heads);
		assert(qk.dim(2) == height * width);
		assert(qk.dim(3) == height * width);

		assert(weights.dim(1) == weights.dim(2));
		const int range = (weights.dim(1) - 1) / 2;

		Tensor t = qk.view( { batch_size, num_heads, height, width, height, width });
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++)
				for (int h1 = 0; h1 < height; h1++)
					for (int w1 = 0; w1 < width; w1++)
					{
						float max_value = -1.0e+32f;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const int offset_h = range + clamp(h2 - h1, -range, range);
								const int offset_w = range + clamp(w2 - w1, -range, range);
								const float bias = weights.get( { h, offset_h, offset_w });
								const float tmp = t.get( { b, h, h1, w1, h2, w2 }) + bias;
								max_value = std::max(max_value, tmp);
								t.set(tmp, { b, h, h1, w1, h2, w2 });
							}

						float sum = 0.0f;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const float tmp = std::exp(t.get( { b, h, h1, w1, h2, w2 }) - max_value);
								sum += tmp;
								t.set(tmp, { b, h, h1, w1, h2, w2 });
							}

						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
								t.set(t.get( { b, h, h1, w1, h2, w2 }) / sum, { b, h, h1, w1, h2, w2 });
					}
	}
	Tensor baseline_softmax_backward(const Tensor &output, const Tensor &gradient_next, Tensor &weights_update, const Shape &input_shape)
	{
		assert(output.shape() == gradient_next.shape());
		const int batch_size = input_shape[0];
		const int height = input_shape[1];
		const int width = input_shape[2];
		const int num_heads = weights_update.firstDim();

		const int range = (weights_update.dim(1) - 1) / 2;

		Tensor result(output.shape(), output.dtype(), output.device());
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++)
				for (int h1 = 0; h1 < height; h1++)
					for (int w1 = 0; w1 < width; w1++)
					{
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const int offset_h = range + clamp(h2 - h1, -range, range);
								const int offset_w = range + clamp(w2 - w1, -range, range);

								const float y = output.get( { b, h, h1 * width + w1, h2 * width + w2 });
								const float dy = gradient_next.get( { b, h, h1 * width + w1, h2 * width + w2 }) * y * (1.0f - y);
								result.set(dy, { b, h, h1 * width + w1, h2 * width + w2 });

								weights_update.set(dy + weights_update.get( { h, offset_h, offset_w }), { h, offset_h, offset_w });
							}
					}
		return result;
	}

	Tensor baseline_mha_forward(const Tensor &input, const Tensor &weights)
	{
		const int num_heads = weights.firstDim();
		Tensor Q = baseline_extract_qkv('Q', input);
		Tensor K = baseline_extract_qkv('K', input);
		Tensor V = baseline_extract_qkv('V', input);

		Q = split_heads(Q, num_heads);
		K = split_heads(K, num_heads);
		V = split_heads(V, num_heads);

		const int head_dim = Q.lastDim();
		const float scale = 1.0f / std::sqrt(head_dim);
		Tensor QK_intermediate = baseline_batched_gemm(Q, K, scale, 'n', 't');
		baseline_softmax_forward(QK_intermediate, weights, input.shape());

		Tensor output_intermediate = baseline_batched_gemm(QK_intermediate, V, 1.0f, 'n', 'n');

		return combine_heads(output_intermediate, input.shape());
	}
	void baseline_mha_backward(const Tensor &input, const Tensor &weights, Tensor &gradient_prev, Tensor &gradient_next, Tensor &weights_update)
	{
		const int num_heads = weights.firstDim();
		Tensor Q = baseline_extract_qkv('Q', input);
		Tensor K = baseline_extract_qkv('K', input);
		Tensor V = baseline_extract_qkv('V', input);

		Q = split_heads(Q, num_heads);
		K = split_heads(K, num_heads);
		V = split_heads(V, num_heads);

		const int head_dim = Q.lastDim();
		const float scale = 1.0 / std::sqrt(head_dim);
		Tensor QK_intermediate = baseline_batched_gemm(Q, K, scale, 'n', 't');
		baseline_softmax_forward(QK_intermediate, weights, input.shape());

		Tensor dy = gradient_next.view( { gradient_next.dim(0), gradient_next.dim(1) * gradient_next.dim(2), gradient_next.dim(3) });
		dy = split_heads(dy, num_heads);
		Tensor d_int = baseline_batched_gemm(dy, V, 1.0f, 'n', 't');
		Tensor dV = baseline_batched_gemm(QK_intermediate, dy, 1.0f, 't', 'n');

		const Tensor d_int2 = baseline_softmax_backward(QK_intermediate, d_int, weights_update, input.shape());

		Tensor dQ = baseline_batched_gemm(d_int2, K, 1.0f / scale, 'n', 'n');
		Tensor dK = baseline_batched_gemm(d_int2, Q, 1.0f / scale, 't', 'n');

		dQ = combine_heads(dQ, input.shape());
		dK = combine_heads(dK, input.shape());
		dV = combine_heads(dV, input.shape());

		baseline_insert_qkv('Q', dQ, gradient_prev);
		baseline_insert_qkv('K', dK, gradient_prev);
		baseline_insert_qkv('V', dV, gradient_prev);
	}
}

namespace ml
{
	TEST(TestMultiHeadAttention, forward)
	{
		const int batch_size = 12;
		const int height = 13;
		const int width = 14;
		const int embedding = 56;
		const int num_heads = 4;
		const int range = 5;
		assert(embedding % num_heads == 0);

		Context context(Device::cpu());

		Tensor input( { batch_size, height, width, 3 * embedding }, "float32", context.device());
		Tensor output( { batch_size, height, width, embedding }, "float32", context.device());
		Tensor weights( { num_heads, range * 2 + 1, range * 2 + 1 }, "float32", context.device());
		testing::initForTest(input, 0.0f);
		testing::initForTest(weights, 1.0);

		const Tensor correct_output = baseline_mha_forward(input, weights);

		const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), weights.shape(), false);
		Tensor workspace( { workspace_size }, "float32", context.device());
		multiHeadAttentionForward(context, input, output, weights, workspace);

//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			output.zeroall();

			const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), weights.shape(), false);
			Tensor workspace( { workspace_size }, "float32", context.device());

			multiHeadAttentionForward(context, input, output, weights, workspace);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestMultiHeadAttention, backward)
	{
		const int batch_size = 11;
		const int height = 12;
		const int width = 13;
		const int embedding = 56;
		const int num_heads = 4;
		const int range = 10;
		assert(embedding % num_heads == 0);

		Context context(Device::cpu());

		Tensor input( { batch_size, height, width, 3 * embedding }, "float32", context.device());
		Tensor output( { batch_size, height, width, embedding }, "float32", context.device());
		Tensor weights( { num_heads, range * 2 + 1, range * 2 + 1 }, "float32", context.device());
		testing::initForTest(input, 0.0f);
		testing::initForTest(weights, 1.0);

		Tensor gradient_prev(input.shape(), input.dtype(), input.device());
		Tensor gradient_next(output.shape(), output.dtype(), output.device());
		Tensor weights_update(weights.shape(), "float32", context.device());
		testing::initForTest(gradient_next, 0.0f);
		testing::initForTest(weights_update, 1.0f);

		Tensor correct_gradient_prev(input.shape(), input.dtype(), input.device());
		Tensor correct_weights_update(weights_update.shape(), weights_update.dtype(), weights_update.device());
		testing::initForTest(correct_weights_update, 1.0f);
		baseline_mha_backward(input, weights, correct_gradient_prev, gradient_next, correct_weights_update);

		const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), weights.shape(), true);
		Tensor workspace( { workspace_size }, "float32", context.device());
		multiHeadAttentionBackward(context, input, weights, gradient_prev, gradient_next, weights_update, workspace);

//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			weights.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			weights_update.moveTo(device);
			output.zeroall();
			testing::initForTest(weights_update, 1.0f);

			const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), weights.shape(), true);
			Tensor workspace( { workspace_size }, "float32", context.device());
			multiHeadAttentionBackward(context, input, weights, gradient_prev, gradient_next, weights_update, workspace);

			context.synchronize();
			for (int i = 0; i < weights_update.dim(0); i++)
				for (int j = 0; j < weights_update.dim(1); j++)
					for (int k = 0; k < weights_update.dim(2); k++)
						if (std::abs(weights_update.get( { i, j, k }) - correct_weights_update.get( { i, j, k })) > 1.0e-3f)
						{
							std::cout << i << "," << j << "," << k << " : " << weights_update.get( { i, j, k }) << " vs "
									<< correct_weights_update.get( { i, j, k }) << '\n';
							exit(0);
						}

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-4f);
		}
	}
} /* namespace ml */
