/*
 * test_moe.cpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_TEST_MOE_CPP_
#define BACKEND_TEST_MOE_CPP_

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/layers/Router.hpp>
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

	template<typename T>
	void baseline_gemm(char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, T alpha, T beta)
	{
		assert(A.device().isCPU());
		assert(B.device().isCPU());
		assert(C.device().isCPU());
		for (int m = 0; m < C.dim(0); m++)
			for (int n = 0; n < C.dim(1); n++)
			{
				T tmp = (T) 0;
				if (opA == 'n')
				{
					if (opB == 'n')
					{ // NN
						for (int k = 0; k < A.dim(1); k++)
							tmp += (T) A.at( { m, k }) * (T) B.at( { k, n });
					}
					else
					{ // NT
						for (int k = 0; k < A.dim(1); k++)
							tmp += (T) A.at( { m, k }) * (T) B.at( { n, k });
					}
				}
				else
				{
					if (opB == 'n')
					{ // TN
						for (int k = 0; k < A.dim(0); k++)
							tmp += (T) A.at( { k, m }) * (T) B.at( { k, n });
					}
					else
					{ // TT
						for (int k = 0; k < A.dim(0); k++)
							tmp += (T) A.at( { k, m }) * (T) B.at( { n, k });
					}
				}
				tmp = alpha * tmp;
				if (beta != (T) 0)
					tmp += beta * (T) C.at( { m, n });
				C.at( { m, n }) = tmp;
			}
	}

	template<typename T>
	Tensor baseline_softmax_forward(const Tensor &input)
	{
		assert(input.rank() == 3);
		const int batch_size = input.dim(0);
		const int experts = input.dim(1);
		const int tokens = input.dim(2);

		Tensor result = zeros_like(input);
		for (int b = 0; b < batch_size; b++)
			for (int e = 0; e < experts; e++)
			{
				T max_value = std::numeric_limits<T>::lowest();
				for (int t = 0; t < tokens; t++)
					max_value = std::max(max_value, (T) input.at( { b, e, t }));

				T sum = 0;
				for (int t = 0; t < tokens; t++)
				{
					const T tmp = std::exp((T) input.at( { b, e, t }) - max_value);
					sum += tmp;
					result.at( { b, e, t }) = tmp;
				}

				for (int t = 0; t < tokens; t++)
					result.at( { b, e, t }) = (T) result.at( { b, e, t }) / sum;
			}
		return result;
	}
	template<typename T>
	Tensor baseline_softmax_backward(const Tensor &output, const Tensor &gradient_next)
	{
		assert(output.rank() == 3);
		assert(output.shape() == gradient_next.shape());
		const int batch_size = output.dim(0);
		const int experts = output.dim(1);
		const int tokens = output.dim(2);

		Tensor result = zeros_like(output);
		for (int b = 0; b < batch_size; b++)
			for (int e = 0; e < experts; e++)
			{
				T tmp = 0;
				for (int t = 0; t < tokens; t++)
				{
					const T y = output.at( { b, e, t });
					const T dy = gradient_next.at( { b, e, t });
					tmp += dy * y;
				}
				for (int t = 0; t < tokens; t++)
				{
					const T y = output.at( { b, e, t });
					const T dy = gradient_next.at( { b, e, t });
					const T dx = y * (dy - tmp);
					result.at( { b, e, t }) = dx;
				}
			}
		return result;
	}

	class BaselineRouter: public Layer
	{
			int m_experts = 0;
		public:
			BaselineRouter(int experts) :
					Layer(),
					m_experts(experts)
			{
			}
			Shape getWeightShape() const
			{
				return Shape( { m_experts, getInputShape().lastDim() });
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
				return Shape( { batch_size, m_experts, height * width });
			}
			std::string name() const
			{
				return "BaselineRouter";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["experts"] = m_experts;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineRouter> result = std::make_unique<BaselineRouter>(config["experts"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				const int batch_size = input[0].dim(0);
				const int tokens = input[0].dim(1) * input[0].dim(2);
				const int channels = input[0].dim(3);

				for (int b = 0; b < batch_size; b++)
				{
					const Tensor x = input[0].view( { tokens, channels }, b * tokens * channels);
					Tensor y = output.view( { m_experts, tokens }, b * m_experts * tokens);
					if (dtype() == DataType::FLOAT32)
						baseline_gemm<float>('n', 't', y, getWeights().getParam(), x, 1.0f, 0.0f);
					else
						baseline_gemm<double>('n', 't', y, getWeights().getParam(), x, 1.0f, 0.0f);
				}
				if (dtype() == DataType::FLOAT32)
					output = baseline_softmax_forward<float>(output);
				else
					output = baseline_softmax_forward<double>(output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (dtype() == DataType::FLOAT32)
					gradient_next = baseline_softmax_backward<float>(output, gradient_next);
				else
					gradient_next = baseline_softmax_backward<double>(output, gradient_next);

				const int batch_size = input[0].dim(0);
				const int tokens = input[0].dim(1) * input[0].dim(2);
				const int channels = input[0].dim(3);
				getWeights().getGradient().zeroall();
				for (int b = 0; b < batch_size; b++)
				{
					const Tensor x = input[0].view( { tokens, channels }, b * tokens * channels);
					Tensor dx = gradient_prev[0].view( { tokens, channels }, b * tokens * channels);
					Tensor dy = gradient_next.view( { m_experts, tokens }, b * m_experts * tokens);

					if (dtype() == DataType::FLOAT32)
					{
						baseline_gemm<float>('t', 'n', dx, dy, getWeights().getParam(), 1.0f, 0.0f);
						baseline_gemm<float>('n', 'n', getWeights().getGradient(), dy, x, 1.0f, 1.0f);
					}
					else
					{
						baseline_gemm<double>('t', 'n', dx, dy, getWeights().getParam(), 1.0f, 0.0f);
						baseline_gemm<double>('n', 'n', getWeights().getGradient(), dy, x, 1.0f, 1.0f);
					}
				}
			}
	};
}

namespace ml
{
//	TEST(TestRouter, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineRouter(5) };
//		gradcheck.setInputShape(Shape( { 3, 8, 10, 37 }));
//
//		gradcheck.check(100, 1.0e-3, "all", true);
//
//		exit(0);
//	}

	TEST(TestRouter, forward)
	{
		const int batch_size = 3;
		const int height = 13;
		const int width = 14;
		const int channels = 56;
		const int experts = 4;

		std::shared_ptr<Context> context = std::make_shared<Context>(Device::cpu());

		Tensor input( { batch_size, height, width, channels }, "float32", context->device());
		Tensor output( { batch_size, experts, height * width }, "float32", context->device());
		std::vector<Tensor> inputs = { input };
		testing::initForTest(input, 0.0);

		Tensor correct_output = zeros_like(output);
		BaselineRouter baseline_router(experts);
		baseline_router.setInputShape( { input.shape() });
		baseline_router.changeContext(context);
		baseline_router.init();

		Router router(experts);
		router.setInputShape( { input.shape() });
		router.changeContext(context);
		router.getWeights().getParam().copyFrom(*context, baseline_router.getWeights().getParam());

		baseline_router.forward(inputs, correct_output);
		router.forward(inputs, output);

		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			std::shared_ptr<Context> gpu_context = std::make_shared<Context>(device);

			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			router.changeContext(gpu_context);

			router.forward(inputs, output);
			gpu_context->synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
		}
	}
	TEST(TestRouter, backward)
	{
		const int batch_size = 3;
		const int height = 13;
		const int width = 14;
		const int channels = 56;
		const int experts = 4;

		std::shared_ptr<Context> context = std::make_shared<Context>(Device::cpu());

		Tensor input( { batch_size, height, width, channels }, "float32", context->device());
		Tensor gradient_prev = zeros_like(input);
		Tensor correct_gradient_prev = zeros_like(input);
		Tensor output( { batch_size, experts, height * width }, "float32", context->device());
		Tensor gradient_next = zeros_like(output);
		std::vector<Tensor> inputs = { input };
		std::vector<Tensor> gradient_prevs = { gradient_prev };
		std::vector<Tensor> correct_gradient_prevs = { correct_gradient_prev };
		std::vector<float> betas = { 0.1f };

		testing::initForTest(input, 0.0);
		testing::initForTest(gradient_next, 0.0);

		Tensor correct_output = zeros_like(output);
		BaselineRouter baseline_router(experts);
		baseline_router.setInputShape( { input.shape() });
		baseline_router.changeContext(context);
		baseline_router.init();
		baseline_router.forward(inputs, output);
		baseline_router.backward(inputs, output, correct_gradient_prevs, gradient_next, betas);

		Router router(experts);
		router.setInputShape( { input.shape() });
		router.changeContext(context);
		router.getWeights().getParam().copyFrom(*context, baseline_router.getWeights().getParam());
		std::shared_ptr<Tensor> workspace = std::make_shared<Tensor>(Shape( { router.getWorkspaceSize() }), router.dtype(), router.device());
		router.setWorkspace(workspace);

		router.backward(inputs, output, gradient_prevs, gradient_next, betas);

		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-4f);
		EXPECT_LE(testing::diffForTest(baseline_router.getWeights().getGradient(), router.getWeights().getGradient()), 1.0e-4f);

//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			std::shared_ptr<Context> gpu_context = std::make_shared<Context>(device);
//			testing::initForTest(gradient_next, 0.0);
//
//			input.moveTo(device);
//			output.moveTo(device);
//			router.changeContext(gpu_context);
//
//			router.forward(inputs, output);
//			gpu_context->synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4f);
//		}
	}

} /* namespace ml */

#endif /* BACKEND_TEST_MOE_CPP_ */
