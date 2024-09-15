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
#include <minml/layers/Layer.hpp>
#include <minml/utils/json.hpp>
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
	int round_up(int x, int y) noexcept
	{
		return y * ((x + y - 1) / y);
	}

	Tensor baseline_extract_qkv(char c, const Tensor &qkv)
	{
		assert(qkv.dtype() == DataType::FLOAT32 || qkv.dtype() == DataType::FLOAT64);
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
				{
					if (qkv.dtype() == DataType::FLOAT32)
						result.at( { b, t, c }) = (float) tmp.at( { b, t, index, c });
					else
						result.at( { b, t, c }) = (double) tmp.at( { b, t, index, c });
				}
		return result;
	}
	void baseline_insert_qkv(char c, const Tensor &x, Tensor &qkv)
	{
		assert(qkv.dtype() == DataType::FLOAT32 || qkv.dtype() == DataType::FLOAT64);
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
				{
					if (qkv.dtype() == DataType::FLOAT32)
						dst.at( { b, t, index, c }) = (float) tmp.at( { b, t, c });
					else
						dst.at( { b, t, index, c }) = (double) tmp.at( { b, t, c });
				}
	}
	Tensor split_heads(const Tensor &x, int num_heads)
	{
		assert(x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT64);
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
					{
						if (x.dtype() == DataType::FLOAT32)
							result.at( { b, h, t, c }) = (float) tmp.at( { b, t, h, c });
						else
							result.at( { b, h, t, c }) = (double) tmp.at( { b, t, h, c });
					}
		return result;
	}
	Tensor combine_heads(const Tensor &x, const Shape &input_shape)
	{
		assert(x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT64);
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
					{
						if (x.dtype() == DataType::FLOAT32)
							result.at( { b, t, h, c }) = (float) x.at( { b, h, t, c });
						else
							result.at( { b, t, h, c }) = (double) x.at( { b, h, t, c });
					}
		result.reshape( { B, input_shape.dim(1), input_shape.dim(2), H * C });
		return result;
	}
	template<typename T>
	Tensor baseline_batched_gemm(const Tensor &A, const Tensor &B, T alpha, char opA, char opB)
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
						T tmp = 0;
						if (opA == 'n')
						{
							if (opB == 'n')
								for (int k = 0; k < K; k++)
									tmp += (T) A.at( { b, h, m, k }) * (T) B.at( { b, h, k, n });
							else
								for (int k = 0; k < K; k++)
									tmp += (T) A.at( { b, h, m, k }) * (T) B.at( { b, h, n, k });
						}
						else
						{
							if (opB == 'n')
								for (int k = 0; k < K; k++)
									tmp += (T) A.at( { b, h, k, m }) * (T) B.at( { b, h, k, n });
							else
								for (int k = 0; k < K; k++)
									tmp += (T) A.at( { b, h, k, m }) * (T) B.at( { b, h, n, k });
						}
						result.at( { b, h, m, n }) = alpha * tmp;
					}
		return result;
	}
	template<typename T>
	void baseline_softmax_forward(Tensor &qk, const Tensor &weights, const Shape &input_shape)
	{
		const int batch_size = input_shape[0];
		const int height = input_shape[1];
		const int width = input_shape[2];
		const int num_heads = weights.firstDim();
		assert(qk.dim(1) == num_heads);
		assert(qk.dim(2) == height * width);
		assert(qk.dim(3) == height * width);

		const int range = (weights.dim(1) - 1) / 2;

		Tensor tmp_qk = qk.view( { batch_size, num_heads, height, width, height, width });
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++)
				for (int h1 = 0; h1 < height; h1++)
					for (int w1 = 0; w1 < width; w1++)
					{
						T max_value = -1.0e+32;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const int offset_h = range + clamp(h2 - h1, -range, range);
								const int offset_w = range + clamp(w2 - w1, -range, range);
								const T bias = weights.at( { h, offset_h, offset_w });
								const T tmp = (T) tmp_qk.at( { b, h, h1, w1, h2, w2 }) + bias;
								max_value = std::max(max_value, tmp);
								tmp_qk.at( { b, h, h1, w1, h2, w2 }) = tmp;
							}

						T sum = 0;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const T tmp = std::exp((T) tmp_qk.at( { b, h, h1, w1, h2, w2 }) - max_value);
								sum += tmp;
								tmp_qk.at( { b, h, h1, w1, h2, w2 }) = tmp;
							}

						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
								tmp_qk.at( { b, h, h1, w1, h2, w2 }) = (T) tmp_qk.at( { b, h, h1, w1, h2, w2 }) / sum;
					}
	}
	template<typename T>
	Tensor baseline_softmax_backward(const Tensor &output, const Tensor &gradient_next, Tensor &weights_update, const Shape &input_shape)
	{
		assert(output.shape() == gradient_next.shape());
		const int batch_size = input_shape[0];
		const int height = input_shape[1];
		const int width = input_shape[2];
		const int num_heads = weights_update.firstDim();

		const int range = (weights_update.dim(1) - 1) / 2;

		Tensor tmp_out = output.view( { batch_size, num_heads, height, width, height, width });
		Tensor tmp_grad = gradient_next.view( { batch_size, num_heads, height, width, height, width });
		Tensor result = zeros_like(tmp_out);
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++)
				for (int h1 = 0; h1 < height; h1++)
					for (int w1 = 0; w1 < width; w1++)
					{
						T tmp = 0;
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const T y = tmp_out.at( { b, h, h1, w1, h2, w2 });
								const T dy = tmp_grad.at( { b, h, h1, w1, h2, w2 });
								tmp += dy * y;
							}
						for (int h2 = 0; h2 < height; h2++)
							for (int w2 = 0; w2 < width; w2++)
							{
								const int offset_h = range + clamp(h2 - h1, -range, range);
								const int offset_w = range + clamp(w2 - w1, -range, range);

								const T y = tmp_out.at( { b, h, h1, w1, h2, w2 });
								const T dy = tmp_grad.at( { b, h, h1, w1, h2, w2 });
								const T dx = y * (dy - tmp);
								result.at( { b, h, h1, w1, h2, w2 }) = dx;

								weights_update.at( { h, offset_h, offset_w }) = dx + (T) weights_update.at( { h, offset_h, offset_w });
							}
					}
		result.reshape(output.shape());
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
		Tensor output_intermediate;
		if (input.dtype() == DataType::FLOAT32)
		{
			Tensor QK_intermediate = baseline_batched_gemm<float>(Q, K, scale, 'n', 't');
			baseline_softmax_forward<float>(QK_intermediate, weights, input.shape());
			output_intermediate = baseline_batched_gemm<float>(QK_intermediate, V, 1.0f, 'n', 'n');
		}
		else
		{
			Tensor QK_intermediate = baseline_batched_gemm<double>(Q, K, scale, 'n', 't');
			baseline_softmax_forward<double>(QK_intermediate, weights, input.shape());
			output_intermediate = baseline_batched_gemm<double>(QK_intermediate, V, 1.0, 'n', 'n');
		}

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
		Tensor QK_intermediate;
		if (input.dtype() == DataType::FLOAT32)
		{
			QK_intermediate = baseline_batched_gemm<float>(Q, K, scale, 'n', 't');
			baseline_softmax_forward<float>(QK_intermediate, weights, input.shape());
		}
		else
		{
			QK_intermediate = baseline_batched_gemm<double>(Q, K, scale, 'n', 't');
			baseline_softmax_forward<double>(QK_intermediate, weights, input.shape());
		}

		Tensor dy = gradient_next.view( { gradient_next.dim(0), gradient_next.dim(1) * gradient_next.dim(2), gradient_next.dim(3) });
		dy = split_heads(dy, num_heads);

		Tensor dQ, dK, dV;
		if (input.dtype() == DataType::FLOAT32)
		{
			Tensor d_int = baseline_batched_gemm<float>(dy, V, 1.0f, 'n', 't');
			dV = baseline_batched_gemm<float>(QK_intermediate, dy, 1.0f, 't', 'n');

			const Tensor d_int2 = baseline_softmax_backward<float>(QK_intermediate, d_int, weights_update, input.shape());

			dQ = baseline_batched_gemm<float>(d_int2, K, scale, 'n', 'n');
			dK = baseline_batched_gemm<float>(d_int2, Q, scale, 't', 'n');
		}
		else
		{
			Tensor d_int = baseline_batched_gemm<double>(dy, V, 1.0, 'n', 't');
			dV = baseline_batched_gemm<double>(QK_intermediate, dy, 1.0, 't', 'n');

			const Tensor d_int2 = baseline_softmax_backward<double>(QK_intermediate, d_int, weights_update, input.shape());

			dQ = baseline_batched_gemm<double>(d_int2, K, scale, 'n', 'n');
			dK = baseline_batched_gemm<double>(d_int2, Q, scale, 't', 'n');
		}

		dQ = combine_heads(dQ, input.shape());
		dK = combine_heads(dK, input.shape());
		dV = combine_heads(dV, input.shape());

		baseline_insert_qkv('Q', dQ, gradient_prev);
		baseline_insert_qkv('K', dK, gradient_prev);
		baseline_insert_qkv('V', dV, gradient_prev);
	}

	class BaselineMHA: public Layer
	{
			int m_number_of_heads = 0;
			int m_positional_encoding_range = 0;
		public:
			BaselineMHA(int numberOfHeads, int positional_encoding_range) :
					Layer(),
					m_number_of_heads(numberOfHeads),
					m_positional_encoding_range(positional_encoding_range)
			{
			}
			Shape getWeightShape() const
			{
				const int tmp = 2 * m_positional_encoding_range - 1;
				return Shape( { m_number_of_heads, tmp, round_up(tmp, 4) });
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int height = getInputShape().dim(1);
				const int width = getInputShape().dim(2);
				const int embedding = getInputShape().dim(3) / 3;
				return Shape( { batch_size, height, width, embedding });
			}
			std::string name() const
			{
				return "BaselineMHA";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["number_of_heads"] = m_number_of_heads;
				result["positional_encoding_range"] = m_positional_encoding_range;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineMHA> result = std::make_unique<BaselineMHA>(config["number_of_heads"].getInt(),
						config["positional_encoding_range"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				output = baseline_mha_forward(input[0], getWeights().getParam());
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
			{
				baseline_mha_backward(input[0], getWeights().getParam(), gradient_prev[0], gradient_next, getWeights().getGradient());
			}
	};
}

namespace ml
{
//	TEST(TestMultiHeadAttention, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineMHA(1, 8) };
//		gradcheck.setInputShape(Shape( { 3, 8, 8, 3 * 32 }));
//
//		gradcheck.check(100, 1.0e-4, "input");
//
//		exit(0);
//	}

	TEST(TestMultiHeadAttention, forward)
	{
		const int batch_size = 3;
		const int height = 13;
		const int width = 14;
		const int embedding = 56;
		const int num_heads = 4;
		const int range = std::max(height, width);
		assert(embedding % num_heads == 0);

		Context context(Device::cpu());

		Tensor input( { batch_size, height, width, 3 * embedding }, "float32", context.device());
		Tensor output( { batch_size, height, width, embedding }, "float32", context.device());
		Tensor weights( { num_heads, range * 2 - 1, round_up(range * 2 - 1, 4) }, "float32", context.device());
		Tensor backward_data;
		testing::initForTest(input, 0.0f);
		testing::initForTest(weights, 1.0);

		const Tensor correct_output = baseline_mha_forward(input, weights);

		const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), weights.shape(), false);
		Tensor workspace( { workspace_size }, "float32", context.device());
		multiHeadAttentionForward(context, input, output, weights, workspace, backward_data);

		for (int i = 0; i < correct_output.dim(0); i++)
			for (int j = 0; j < correct_output.dim(1); j++)
				for (int k = 0; k < correct_output.dim(2); k++)
					for (int l = 0; l < correct_output.dim(3); l++)
						if (std::fabs(correct_output.get( { i, j, k, l }) - output.get( { i, j, k, l })) > 1.0e-1f)
						{
							std::cout << i << "," << j << "," << k << "," << l << "," << " : " << correct_output.get( { i, j, k, l }) << " vs "
									<< output.get( { i, j, k, l }) << '\n';
							exit(255);
						}
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3f);

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

			multiHeadAttentionForward(context, input, output, weights, workspace, backward_data);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestMultiHeadAttention, backward)
	{
		const int batch_size = 3;
		const int height = 13;
		const int width = 14;
		const int embedding = 56;
		const int num_heads = 4;
		const int range = std::max(height, width);
		assert(embedding % num_heads == 0);

		Context context(Device::cpu());

		Tensor input( { batch_size, height, width, 3 * embedding }, "float32", context.device());
		Tensor output( { batch_size, height, width, embedding }, "float32", context.device());
		Tensor weights( { num_heads, range * 2 - 1, round_up(range * 2 - 1, 4) }, "float32", context.device());
		Tensor backward_data( { batch_size, num_heads, height * width, height * width }, "float32", context.device());
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

		Tensor target(output.shape(), output.dtype(), output.device());
		testing::initForTest(target, 0.0f);

//		const float eps = 1.0e-3f;
//		std::initializer_list<int> index = { 0, 0, 0, 0 };
//
//		Tensor output_0 = baseline_mha_forward(input, weights);
//		Tensor grad_0 = l2_grad(output_0, target);
//		gradient_prev.zeroall();
//		multiHeadAttentionBackward(context, input, weights, gradient_prev, grad_0, weights_update, workspace);
//		context.synchronize();
//		baseline_mha_backward(input, weights, gradient_prev, grad_0, weights_update);
//
//		input.set(input.get(index) + eps, index);
//		const Tensor output_p = baseline_mha_forward(input, weights);
//		const float loss_p = l2_loss(output_p, target);
//
//		input.set(input.get(index) - 2 * eps, index);
//		const Tensor output_m = baseline_mha_forward(input, weights);
//		const float loss_m = l2_loss(output_m, target);
//
//		std::cout << loss_p << " " << loss_m << '\n';
//		std::cout << testing::diffForTest(output_p, output_m) << '\n';
//		std::cout << "backprop grad = " << gradient_prev.get(index) << " vs numerical = " << (loss_p - loss_m) / (2 * eps) << '\n';
//		exit(0);

		multiHeadAttentionForward(context, input, output, weights, workspace, backward_data);
		multiHeadAttentionBackward(context, input, weights, gradient_prev, gradient_next, weights_update, workspace, backward_data);

		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-4f);

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
			backward_data.moveTo(device);
			output.zeroall();
			backward_data.zeroall();
			testing::initForTest(weights_update, 1.0f);

			const int workspace_size = multiHeadAttentionGetWorkspaceSize(context, input.shape(), weights.shape(), true);
			Tensor workspace( { workspace_size }, "float32", context.device());
			multiHeadAttentionForward(context, input, output, weights, workspace, backward_data);
			multiHeadAttentionBackward(context, input, weights, gradient_prev, gradient_next, weights_update, workspace, backward_data);

			context.synchronize();
//			for (int i = 0; i < weights_update.dim(0); i++)
//				for (int j = 0; j < weights_update.dim(1); j++)
//					for (int k = 0; k < weights_update.dim(2); k++)
//						if (std::abs(weights_update.get( { i, j, k }) - correct_weights_update.get( { i, j, k })) > 1.0e-3f)
//						{
//							std::cout << i << "," << j << "," << k << " : " << weights_update.get( { i, j, k }) << " vs "
//									<< correct_weights_update.get( { i, j, k }) << '\n';
//							exit(0);
//						}

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
			EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-4f);
		}
	}
} /* namespace ml */
