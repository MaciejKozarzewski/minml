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

	Tensor baseline_extract_qkv(char c, const Tensor &qkv)
	{
		assert(qkv.rank() == 3);
		assert(qkv.dim(2) % 3 == 0);
		const int B = qkv.dim(0);
		const int T = qkv.dim(1);
		const int C = qkv.dim(2) / 3;
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
		assert(qkv.rank() == 3);
		assert(qkv.dim(2) % 3 == 0);
		const int B = qkv.dim(0);
		const int T = qkv.dim(1);
		const int C = qkv.dim(2) / 3;
		Tensor dst = qkv.view( { B, T, 3, C });

		const int index = (c == 'Q') ? 0 : ((c == 'K') ? 1 : 2);

		for (int b = 0; b < B; b++)
			for (int t = 0; t < T; t++)
				for (int c = 0; c < C; c++)
					dst.set(x.get( { b, t, c }), { b, t, index, c });
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
	Tensor combine_heads(const Tensor &x)
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
		result.reshape( { B, T, H * C });
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
	void baseline_softmax_in_place(Tensor &qk)
	{
		const int first_dim = qk.shape().volumeWithoutLastDim();
		const int last_dim = qk.lastDim();

		Tensor t = qk.view( { first_dim, last_dim });
		for (int i = 0; i < first_dim; i++)
		{
			float max_value = -1.0e+32f;
			for (int j = 0; j < last_dim; j++)
				max_value = std::max(max_value, t.get( { i, j }));

			float sum = 0.0f;
			for (int j = 0; j < last_dim; j++)
			{
				const float tmp = std::exp(t.get( { i, j }) - max_value);
				sum += tmp;
				t.set(tmp, { i, j });
			}

			for (int j = 0; j < last_dim; j++)
				t.set(t.get( { i, j }) / sum, { i, j });
		}
	}
	Tensor baseline_softmax_backward(const Tensor &output, const Tensor &gradient_next)
	{
		assert(output.shape() == gradient_next.shape());
		const int first_dim = output.shape().volumeWithoutLastDim();
		const int last_dim = output.lastDim();

		const Tensor out = output.view( { first_dim, last_dim });
		const Tensor next = gradient_next.view( { first_dim, last_dim });
		Tensor result( { first_dim, last_dim }, output.dtype(), output.device());
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				const float y = out.get( { i, j });
				const float dy = next.get( { i, j });
				result.set(dy * y * (1.0f - y), { i, j });
			}
		result.reshape(output.shape());
		return result;
	}

	Tensor baseline_mha_forward(const Tensor &input, int num_heads)
	{
		Tensor Q = baseline_extract_qkv('Q', input);
		Tensor K = baseline_extract_qkv('K', input);
		Tensor V = baseline_extract_qkv('V', input);

		Q = split_heads(Q, num_heads);
		K = split_heads(K, num_heads);
		V = split_heads(V, num_heads);

		const int head_dim = Q.lastDim();
		const float scale = 1.0f / std::sqrt(head_dim);
		Tensor QK_intermediate = baseline_batched_gemm(Q, K, scale, 'n', 't');
		baseline_softmax_in_place(QK_intermediate);

		Tensor output_intermediate = baseline_batched_gemm(QK_intermediate, V, 1.0f, 'n', 'n');

		return combine_heads(output_intermediate);
	}
	void baseline_mha_backward(const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, int num_heads)
	{
		Tensor Q = baseline_extract_qkv('Q', input);
		Tensor K = baseline_extract_qkv('K', input);
		Tensor V = baseline_extract_qkv('V', input);

		Q = split_heads(Q, num_heads);
		K = split_heads(K, num_heads);
		V = split_heads(V, num_heads);

		const int head_dim = Q.lastDim();
		const float scale = 1.0 / std::sqrt(head_dim);
		Tensor QK_intermediate = baseline_batched_gemm(Q, K, scale, 'n', 't');
		baseline_softmax_in_place(QK_intermediate);

		const Tensor dy = split_heads(gradient_next, num_heads);
		Tensor d_int = baseline_batched_gemm(dy, V, 1.0f, 'n', 't');
		Tensor dV = baseline_batched_gemm(QK_intermediate, dy, 1.0f, 't', 'n');

		const Tensor d_int2 = d_int;//baseline_softmax_backward(QK_intermediate, d_int);

		Tensor dQ = baseline_batched_gemm(d_int2, K, 1.0f, 'n', 'n');
		Tensor dK = baseline_batched_gemm(d_int2, Q, 1.0f, 't', 'n');

		dQ = combine_heads(dQ);
		dK = combine_heads(dK);
		dV = combine_heads(dV);

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
		const int tokens = 34;
		const int embedding = 56;
		const int num_heads = 4;
		assert(embedding % num_heads == 0);

		Context context(Device::cpu());

		Tensor input( { batch_size, tokens, 3 * embedding }, "float32", context.device());
		Tensor output( { batch_size, tokens, embedding }, "float32", context.device());
		testing::initForTest(input, 0.0f);

		const Tensor correct_output = baseline_mha_forward(input, num_heads);

		const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), num_heads, false);
		Tensor workspace( { workspace_size }, "float32", context.device());
		multiHeadAttentionForward(context, input, output, num_heads, workspace);

//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();

			const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), num_heads, false);
			Tensor workspace( { workspace_size }, "float32", context.device());

			multiHeadAttentionForward(context, input, output, num_heads, workspace);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestMultiHeadAttention, backward)
	{
		const int batch_size = 12;
		const int tokens = 34;
		const int embedding = 56;
		const int num_heads = 4;
		assert(embedding % num_heads == 0);

		Context context(Device::cpu());

		Tensor input( { batch_size, tokens, 3 * embedding }, "float32", context.device());
		Tensor output( { batch_size, tokens, embedding }, "float32", context.device());

		Tensor gradient_prev(input.shape(), input.dtype(), input.device());
		Tensor gradient_next(output.shape(), output.dtype(), output.device());
		testing::initForTest(input, 0.0f);
		testing::initForTest(gradient_next, 1.0f);

		Tensor correct_gradient_prev(input.shape(), input.dtype(), input.device());
		baseline_mha_backward(input, correct_gradient_prev, gradient_next, num_heads);

		const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), num_heads, true);
		Tensor workspace( { workspace_size }, "float32", context.device());
		multiHeadAttentionBackward(context, input, gradient_prev, gradient_next, num_heads, workspace);

//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			output.zeroall();

			const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), num_heads, true);
			Tensor workspace( { workspace_size }, "float32", context.device());
			multiHeadAttentionBackward(context, input, gradient_prev, gradient_next, num_heads, workspace);

			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		}
	}
} /* namespace ml */
