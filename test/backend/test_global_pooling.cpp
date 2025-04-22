/*
 * test_global_pooling.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/layers/LearnableGlobalPooling.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/json.hpp>

#include <cmath>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	template<typename T = float>
	void baseline_average_pooling_forward(const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 4);
		assert(output.rank() == 2);
		assert(input.firstDim() == output.firstDim());
		assert(input.lastDim() == output.lastDim());

		const T inv = 1.0 / (input.dim(1) * input.dim(2));
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
			{
				T avg_value = 0.0;
				for (int j = 0; j < input.dim(1); j++)
					for (int k = 0; k < input.dim(2); k++)
						avg_value += (T) input.at( { i, j, k, l });
				output.at( { i, l }) = avg_value * inv;
			}
	}
	template<typename T = float>
	void baseline_average_pooling_backward(Tensor &gradient_prev, const Tensor &gradient_next)
	{
		assert(gradient_prev.rank() == 4);
		assert(gradient_next.rank() == 2);
		assert(gradient_prev.firstDim() == gradient_next.firstDim());
		assert(gradient_prev.lastDim() == gradient_next.lastDim());

		const T inv = 1.0 / (gradient_prev.dim(1) * gradient_prev.dim(2));
		for (int i = 0; i < gradient_prev.dim(0); i++)
			for (int l = 0; l < gradient_prev.dim(3); l++)
				for (int j = 0; j < gradient_prev.dim(1); j++)
					for (int k = 0; k < gradient_prev.dim(2); k++)
						gradient_prev.at( { i, j, k, l }) = inv * (T) gradient_next.at( { i, l });
	}

	template<typename T = float>
	void baseline_learnable_pooling_forward(const Tensor &input, const Tensor &weights, Tensor &output)
	{
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
				for (int out = 0; out < weights.firstDim(); out++)
				{
					T acc = 0.0;
					int m = 0;
					for (int j = 0; j < input.dim(1); j++)
						for (int k = 0; k < input.dim(2); k++, m++)
							acc += (T) input.at( { i, j, k, l }) * (T) weights.at( { out, m });
					output.at( { i, l * weights.firstDim() + out }) = acc;
				}
	}
	template<typename T = float>
	void baseline_learnable_pooling_backward(Tensor &gradient_prev, const Tensor &weights, const Tensor &gradient_next)
	{
		gradient_prev.zeroall();
		for (int i = 0; i < gradient_prev.firstDim(); i++)
			for (int l = 0; l < gradient_prev.lastDim(); l++)
				for (int out = 0; out < weights.firstDim(); out++)
				{
					const T grad = (T) gradient_next.at( { i, l * weights.firstDim() + out });
					int m = 0;
					for (int j = 0; j < gradient_prev.dim(1); j++)
						for (int k = 0; k < gradient_prev.dim(2); k++, m++)
							gradient_prev.at( { i, j, k, l }) = (T) gradient_prev.at( { i, j, k, l }) + grad * (T) weights.at( { out, m });
				}
	}
	template<typename T = float>
	void baseline_learnable_pooling_update(const Tensor &input, Tensor &weights_update, const Tensor &gradient_next)
	{
		weights_update.zeroall();
		for (int i = 0; i < input.firstDim(); i++)
			for (int l = 0; l < input.lastDim(); l++)
				for (int out = 0; out < weights_update.firstDim(); out++)
				{
					const T grad = (T) gradient_next.at( { i, l * weights_update.firstDim() + out });
					int m = 0;
					for (int j = 0; j < input.dim(1); j++)
						for (int k = 0; k < input.dim(2); k++, m++)
							weights_update.at( { out, m }) = (T) weights_update.at( { out, m }) + grad * (T) input.at( { i, j, k, l });
				}
	}

	template<typename T = float>
	void baseline_channel_scaling_forward(const Tensor &input, Tensor &output, const Tensor &scales)
	{
		assert(input.rank() == 4);
		assert(input.shape() == output.shape());
		assert(scales.rank() == 2);
		assert(scales.firstDim() == output.firstDim());
		assert(scales.lastDim() == output.lastDim());

		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					for (int l = 0; l < input.dim(3); l++)
						output.at( { i, j, k, l }) = (T) input.at( { i, j, k, l }) * (T) scales.at( { i, l });
	}
	template<typename T = float>
	void baseline_channel_scaling_backward(Tensor &gradient_prev, Tensor &gradient_scales, const Tensor &gradient_next, const Tensor &input,
			const Tensor &scales)
	{
		assert(input.rank() == 4);
		assert(gradient_prev.shape() == input.shape());
		assert(gradient_scales.shape() == scales.shape());
		assert(gradient_prev.shape() == gradient_next.shape());

		for (int i = 0; i < gradient_next.dim(0); i++)
			for (int l = 0; l < gradient_next.dim(3); l++)
			{
				T grad = 0;
				for (int j = 0; j < gradient_next.dim(1); j++)
					for (int k = 0; k < gradient_next.dim(2); k++)
					{
						grad += (T) gradient_next.at( { i, j, k, l }) * (T) input.at( { i, j, k, l });
						gradient_prev.at( { i, j, k, l }) = (T) scales.at( { i, l }) * (T) gradient_next.at( { i, j, k, l });
					}
				gradient_scales.at( { i, l }) = grad;
			}
	}

	template<typename T = float>
	void baseline_channel_average_pooling_forward(const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 4);
		assert(output.rank() == 2);

		const T inv = 1.0 / input.dim(3);
		for (int i = 0; i < input.firstDim(); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
				{
					T avg_value = 0.0;
					for (int l = 0; l < input.lastDim(); l++)
						avg_value += (T) input.at( { i, j, k, l });
					output.at( { i, j * input.dim(2) + k }) = avg_value * inv;
				}
	}
	template<typename T = float>
	void baseline_channel_average_pooling_backward(Tensor &gradient_prev, const Tensor &gradient_next)
	{
		assert(gradient_prev.rank() == 4);
		assert(gradient_next.rank() == 2);

		const T inv = 1.0 / gradient_prev.dim(3);
		for (int i = 0; i < gradient_prev.dim(0); i++)
			for (int j = 0; j < gradient_prev.dim(1); j++)
				for (int k = 0; k < gradient_prev.dim(2); k++)
					for (int l = 0; l < gradient_prev.dim(3); l++)
						gradient_prev.at( { i, j, k, l }) = inv * (T) gradient_next.at( { i, j * gradient_prev.dim(2) + k });
	}

	template<typename T = float>
	void baseline_spatial_scaling_forward(const Tensor &input, Tensor &output, const Tensor &scales)
	{
		assert(input.rank() == 4);
		assert(input.shape() == output.shape());
		assert(scales.rank() == 2);

		for (int i = 0; i < input.dim(0); i++)
			for (int j = 0; j < input.dim(1); j++)
				for (int k = 0; k < input.dim(2); k++)
					for (int l = 0; l < input.dim(3); l++)
						output.at( { i, j, k, l }) = (T) input.at( { i, j, k, l }) * (T) scales.at( { i, j * input.dim(2) + k });
	}
	template<typename T = float>
	void baseline_spatial_scaling_backward(Tensor &gradient_prev, Tensor &gradient_scales, const Tensor &gradient_next, const Tensor &input,
			const Tensor &scales)
	{
		assert(input.rank() == 4);
		assert(gradient_prev.shape() == input.shape());
		assert(gradient_scales.shape() == scales.shape());
		assert(gradient_prev.shape() == gradient_next.shape());

		for (int i = 0; i < gradient_next.dim(0); i++)
			for (int j = 0; j < gradient_next.dim(1); j++)
				for (int k = 0; k < gradient_next.dim(2); k++)
				{
					T grad = 0;
					for (int l = 0; l < gradient_next.dim(3); l++)
					{
						grad += (T) gradient_next.at( { i, j, k, l }) * (T) input.at( { i, j, k, l });
						gradient_prev.at( { i, j, k, l }) = (T) scales.at( { i, j * gradient_next.dim(2) + k })
								* (T) gradient_next.at( { i, j, k, l });
					}
					gradient_scales.at( { i, j * gradient_next.dim(2) + k }) = grad;
				}
	}

	class BaselineGlobalAveragePooling: public Layer
	{
		public:
			BaselineGlobalAveragePooling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().firstDim();
				const int channels = getInputShape().lastDim();
				return Shape( { batch_size, channels });
			}
			std::string name() const
			{
				return "BaselineGlobalAveragePooling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineGlobalAveragePooling> result = std::make_unique<BaselineGlobalAveragePooling>();
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_average_pooling_forward<float>(input[0], output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_average_pooling_forward<double>(input[0], output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_average_pooling_backward<float>(gradient_prev[0], gradient_next);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_average_pooling_backward<double>(gradient_prev[0], gradient_next);
			}
	};
	class BaselineLearbableGlobalPooling: public Layer
	{
			int m_dim = 0;
		public:
			BaselineLearbableGlobalPooling(int dim) :
					Layer("linear"),
					m_dim(dim)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().firstDim();
				const int channels = getInputShape().lastDim();
				return Shape( { batch_size, m_dim * channels });
			}
			Shape getWeightShape() const
			{
				return Shape( { m_dim, getInputShape().dim(1) * getInputShape().dim(2) });
			}
			std::string name() const
			{
				return "BaselineLearbableGlobalPooling";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["dim"] = m_dim;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineLearbableGlobalPooling> result = std::make_unique<BaselineLearbableGlobalPooling>(config["dim"].getInt());
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_learnable_pooling_forward<float>(input[0], getWeights().getParam(), output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_learnable_pooling_forward<double>(input[0], getWeights().getParam(), output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
				{
					baseline_learnable_pooling_backward<float>(gradient_prev[0], getWeights().getParam(), gradient_next);
					baseline_learnable_pooling_update<float>(input[0], getWeights().getGradient(), gradient_next);
				}
				if (input[0].dtype() == DataType::FLOAT64)
				{
					baseline_learnable_pooling_backward<double>(gradient_prev[0], getWeights().getParam(), gradient_next);
					baseline_learnable_pooling_update<double>(input[0], getWeights().getGradient(), gradient_next);
				}
			}
	};
	class BaselineChannelScaling: public Layer
	{
		public:
			BaselineChannelScaling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				return getInputShape(0);
			}
			std::string name() const
			{
				return "BaselineChannelScaling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineChannelScaling> result = std::make_unique<BaselineChannelScaling>();
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_channel_scaling_forward<float>(input[0], output, input[1]);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_channel_scaling_forward<double>(input[0], output, input[1]);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_channel_scaling_backward<float>(gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_channel_scaling_backward<double>(gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
			}
	};
	class BaselineChannelAveragePooling: public Layer
	{
		public:
			BaselineChannelAveragePooling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().firstDim();
				const int height = getInputShape().dim(1);
				const int width = getInputShape().dim(2);
				return Shape( { batch_size, height * width });
			}
			std::string name() const
			{
				return "BaselineChannelAveragePooling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineChannelAveragePooling> result = std::make_unique<BaselineChannelAveragePooling>();
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_channel_average_pooling_forward<float>(input[0], output);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_channel_average_pooling_forward<double>(input[0], output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_channel_average_pooling_backward<float>(gradient_prev[0], gradient_next);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_channel_average_pooling_backward<double>(gradient_prev[0], gradient_next);
			}
	};
	class BaselineSpatialScaling: public Layer
	{
		public:
			BaselineSpatialScaling() :
					Layer("linear")
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				return getInputShape(0);
			}
			std::string name() const
			{
				return "BaselineSpatialScaling";
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineSpatialScaling> result = std::make_unique<BaselineSpatialScaling>();
				result->loadConfig(config);
				return result;
			}
			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_spatial_scaling_forward<float>(input[0], output, input[1]);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_spatial_scaling_forward<double>(input[0], output, input[1]);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				if (input[0].dtype() == DataType::FLOAT32)
					baseline_spatial_scaling_backward<float>(gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
				if (input[0].dtype() == DataType::FLOAT64)
					baseline_spatial_scaling_backward<double>(gradient_prev[0], gradient_prev[1], gradient_next, input[0], input[1]);
			}
	};

}

namespace ml
{
//	TEST(TestGlobalAveragePooling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineGlobalAveragePooling() };
//		gradcheck.setInputShape(Shape( { 13, 7, 8, 34 }));
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestLearnableGlobalPooling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineLearbableGlobalPooling(1) };
//		gradcheck.setInputShape(Shape( { 13, 7, 8, 34 }));
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestChannelScaling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineChannelScaling() };
//		std::vector<Shape> shapes = { Shape( { 13, 7, 8, 34 }), Shape( { 13, 34 }) };
//		gradcheck.setInputShape(shapes);
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestChannelAveragePooling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineChannelAveragePooling() };
//		gradcheck.setInputShape(Shape( { 13, 7, 8, 34 }));
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}
//	TEST(TestSpatialScaling, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineSpatialScaling() };
//		std::vector<Shape> shapes = { Shape( { 13, 7, 8, 34 }), Shape( { 13, 7 * 8 }) };
//		gradcheck.setInputShape(shapes);
//
//		gradcheck.check(100, 1.0e-4, "all", true);
//
//		exit(0);
//	}

	TEST(TestGlobalAveragePooling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor output( { input.firstDim(), input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);

		baseline_average_pooling_forward(input, correct_output);

		globalAveragePoolingForward(context, 1.0f, input, 0.0f, output);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalAveragePoolingForward(context, 1.0f, input, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
		}
	}
	TEST(TestGlobalAveragePooling, backward_fp32)
	{
		Context context;
		Tensor gradient_prev( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor gradient_next( { gradient_prev.firstDim(), gradient_prev.lastDim() }, "float32", Device::cpu());

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

		testing::initForTest(gradient_next, 1.57);
		baseline_average_pooling_backward(correct_gradient_prev, gradient_next);

		globalAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			globalAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6);
		}
	}
	TEST(TestGlobalAveragePooling, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor output( { input.firstDim(), input.lastDim() }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);

		baseline_average_pooling_forward(input, correct_output);

		globalAveragePoolingForward(context, 1.0f, input, 0.0f, output);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			globalAveragePoolingForward(context, 1.0f, input, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4);
		}
	}
	TEST(TestGlobalAveragePooling, backward_fp16)
	{
		Context context;
		Tensor gradient_prev( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor gradient_next( { gradient_prev.firstDim(), gradient_prev.lastDim() }, gradient_prev.dtype(), Device::cpu());

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

		testing::initForTest(gradient_next, 1.57);
		baseline_average_pooling_backward(correct_gradient_prev, gradient_next);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			globalAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6);
		}
	}

//	TEST(TestLearnableGlobalPooling, forward_fp32)
//	{
//		Context context;
//		const int batch_size = 12;
//		const int height = 13;
//		const int width = 14;
//		const int channels = 134;
//		const int expansion = 4;
//		Tensor input( { batch_size, height, width, channels });
//		Tensor output( { batch_size, expansion * channels });
//		Tensor weights( { expansion, height * width });
//		testing::initForTest(input, 0.0);
//		testing::initForTest(weights, 1.0);
//
//		Tensor correct_output = zeros_like(output);
//		baseline_learnable_pooling_forward(input, weights, correct_output);
//
//		Tensor tmp_in = input.view( { batch_size, height * width, channels });
//		Tensor tmp_out = output.view( { batch_size, channels, expansion });
//		gemmBatched(context, 't', 't', tmp_out, tmp_in, weights, 1, 0);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			output.moveTo(device);
//			weights.moveTo(device);
//			output.zeroall();
//
//			Tensor tmp_in = input.view( { batch_size, height * width, channels });
//			Tensor tmp_out = output.view( { batch_size, channels, expansion });
//			gemmBatched(context, 't', 't', tmp_out, tmp_in, weights, 1, 0);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
//		}
//	}
//	TEST(TestLearnableGlobalPooling, backward)
//	{
//		Context context;
//		const int batch_size = 12;
//		const int height = 13;
//		const int width = 14;
//		const int channels = 134;
//		const int expansion = 4;
//		Tensor gradient_prev( { batch_size, height, width, channels });
//		Tensor weights( { expansion, height * width });
//		Tensor gradient_next( { batch_size, expansion * channels });
//		testing::initForTest(weights, 0.0);
//		testing::initForTest(gradient_next, 1.57);
//
//		Tensor correct_gradient_prev = zeros_like(gradient_prev);
//		baseline_learnable_pooling_backward(correct_gradient_prev, weights, gradient_next);
//
//		Tensor tmp_prev = gradient_prev.view( { batch_size, height * width, channels });
//		Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//
//		gemmBatched(context, 't', 't', tmp_prev, weights, tmp_next, 1, 0);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			gradient_prev.moveTo(device);
//			weights.moveTo(device);
//			gradient_next.moveTo(device);
//			gradient_prev.zeroall();
//
//			Tensor tmp_prev = gradient_prev.view( { batch_size, height * width, channels });
//			Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//			gemmBatched(context, 't', 't', tmp_prev, weights, tmp_next, 1, 0);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);
//		}
//	}
//	TEST(TestLearnableGlobalPooling, update)
//	{
//		Context context;
//		const int batch_size = 12;
//		const int height = 13;
//		const int width = 14;
//		const int channels = 134;
//		const int expansion = 4;
//		Tensor input( { batch_size, height, width, channels });
//		Tensor gradient_next( { batch_size, expansion * channels });
//		Tensor weights_update( { expansion, height * width });
//		testing::initForTest(input, 0.0);
//		testing::initForTest(gradient_next, 1.57);
//
//		Tensor workspace( { weights_update.volume() * input.firstDim() });
//		Tensor correct_weights_update = zeros_like(weights_update);
//		baseline_learnable_pooling_update(input, correct_weights_update, gradient_next);
//
//		Tensor tmp_in = input.view( { batch_size, height * width, channels });
//		Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//		Tensor tmp_update = workspace.view( { batch_size, expansion, height * width });
//		gemmBatched(context, 't', 't', tmp_update, tmp_next, tmp_in, 1, 0);
////		sumOverFirstDim(context, weights_update, tmp_update.view( { batch_size, expansion * height * width }), 0);
//
//		EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-5f);
//
//		if (testing::has_device_supporting(DataType::FLOAT32))
//		{
//			const Device device = testing::get_device_for_test();
//			Context context(device);
//			input.moveTo(device);
//			gradient_next.moveTo(device);
//			weights_update.moveTo(device);
//			workspace.moveTo(device);
//			weights_update.zeroall();
//
//			Tensor tmp_in = input.view( { batch_size, height * width, channels });
//			Tensor tmp_next = gradient_next.view( { batch_size, channels, expansion });
//			Tensor tmp_update = workspace.view( { batch_size, expansion, height * width });
//			gemmBatched(context, 't', 't', tmp_update, tmp_next, tmp_in, 1, 0);
////			sumOverFirstDim(context, weights_update, tmp_update.view( { batch_size, expansion * height * width }), 0);
//			context.synchronize();
//
//			EXPECT_LE(testing::diffForTest(correct_weights_update, weights_update), 1.0e-5f);
//		}
//	}

	TEST(TestChannelScaling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor scales( { input.firstDim(), input.lastDim() }, "float32", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);

		baseline_channel_scaling_forward(input, correct_output, scales);

		channelScalingForward(context, 1.0f, input, scales, 0.0f, output);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			channelScalingForward(context, 1.0f, input, scales, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
		}
	}
	TEST(TestChannelScaling, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor scales( { input.firstDim(), input.lastDim() }, "float16", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);

		baseline_channel_scaling_forward(input, correct_output, scales);

		channelScalingForward(context, 1.0f, input, scales, 0.0f, output);
		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			channelScalingForward(context, 1.0f, input, scales, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3f);
		}
	}

	TEST(TestChannelScaling, backward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 });
		Tensor scales( { input.firstDim(), input.lastDim() });
		Tensor gradient_next = zeros_like(input);

		Tensor gradient_input = zeros_like(input);
		Tensor gradient_scales = zeros_like(scales);

		Tensor correct_gradient_input = zeros_like(gradient_input);
		Tensor correct_gradient_scales = zeros_like(gradient_scales);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);
		testing::initForTest(gradient_next, 2.0);

		baseline_channel_scaling_backward(correct_gradient_input, correct_gradient_scales, gradient_next, input, scales);

//		channelScalingBackward(context, gradient_input, gradient_scales, gradient_next, input, scales);
//		EXPECT_LE(testing::diffForTest(correct_gradient_input, gradient_input), 1.0e-6f);
//		EXPECT_LE(testing::diffForTest(correct_gradient_scales, gradient_scales), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			gradient_next.moveTo(device);
			gradient_input.moveTo(device);
			gradient_scales.moveTo(device);
			gradient_input.zeroall();
			gradient_scales.zeroall();
			channelScalingBackward(context, 1.0f, gradient_next, input, scales, 0.0f, gradient_input, 0.0f, gradient_scales);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_input, gradient_input), 1.0e-5f);
			EXPECT_LE(testing::diffForTest(correct_gradient_scales, gradient_scales), 1.0e-5f);
		}
	}

	TEST(TestChannelAveragePooling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor output( { input.firstDim(), input.dim(1) * input.dim(2) }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);

		baseline_channel_average_pooling_forward(input, correct_output);

//		channelAveragePoolingForward(context, 1.0f, input, 0.0f, output);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			channelAveragePoolingForward(context, 1.0f, input, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6);
		}
	}
	TEST(TestChannelAveragePooling, backward_fp32)
	{
		Context context;
		Tensor gradient_prev( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor gradient_next( { gradient_prev.firstDim(), gradient_prev.dim(1) * gradient_prev.dim(2) }, "float32", Device::cpu());

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

		testing::initForTest(gradient_next, 1.57);
		baseline_channel_average_pooling_backward(correct_gradient_prev, gradient_next);

//		channelAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
//		EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			channelAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-6);
		}
	}
	TEST(TestChannelAveragePooling, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor output( { input.firstDim(), input.dim(1) * input.dim(2) }, input.dtype(), Device::cpu());
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);

		baseline_channel_average_pooling_forward(input, correct_output);

//		channelAveragePoolingForward(context, 1.0f, input, 0.0f, output);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			channelAveragePoolingForward(context, 1.0f, input, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-4);
		}
	}
	TEST(TestChannelAveragePooling, backward_fp16)
	{
		Context context;
		Tensor gradient_prev( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor gradient_next( { gradient_prev.firstDim(), gradient_prev.dim(1) * gradient_prev.dim(2) }, gradient_prev.dtype(), Device::cpu());

		Tensor correct_gradient_prev = zeros_like(gradient_prev);

		testing::initForTest(gradient_next, 1.57);
		baseline_channel_average_pooling_backward(correct_gradient_prev, gradient_next);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			gradient_prev.moveTo(device);
			gradient_next.moveTo(device);
			gradient_prev.zeroall();
			channelAveragePoolingBackward(context, 1.0f, gradient_next, 0.0f, gradient_prev);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_prev, gradient_prev), 1.0e-5);
		}
	}

	TEST(TestSpatialScaling, forward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float32", Device::cpu());
		Tensor scales( { input.firstDim(), input.dim(1) * input.dim(2) }, "float32", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);

		baseline_spatial_scaling_forward(input, correct_output, scales);

//		spatialScalingForward(context, 1.0f, input, scales, 0.0f, output);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			spatialScalingForward(context, 1.0f, input, scales, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-6f);
		}
	}
	TEST(TestSpatialScaling, forward_fp16)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 }, "float16", Device::cpu());
		Tensor scales( { input.firstDim(), input.dim(1) * input.dim(2) }, "float16", Device::cpu());
		Tensor output = zeros_like(input);
		Tensor correct_output = zeros_like(output);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);

		baseline_spatial_scaling_forward(input, correct_output, scales);

//		spatialScalingForward(context, 1.0f, input, scales, 0.0f, output);
//		EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3f);

		if (testing::has_device_supporting(DataType::FLOAT16))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			output.moveTo(device);
			output.zeroall();
			spatialScalingForward(context, 1.0f, input, scales, 0.0f, output);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_output, output), 1.0e-3f);
		}
	}
	TEST(TestSpatialScaling, backward_fp32)
	{
		Context context;
		Tensor input( { 12, 13, 14, 132 });
		Tensor scales( { input.firstDim(), input.dim(1) * input.dim(2) });
		Tensor gradient_next = zeros_like(input);

		Tensor gradient_input = zeros_like(input);
		Tensor gradient_scales = zeros_like(scales);

		Tensor correct_gradient_input = zeros_like(gradient_input);
		Tensor correct_gradient_scales = zeros_like(gradient_scales);

		testing::initForTest(input, 0.0);
		testing::initForTest(scales, 1.0);
		testing::initForTest(gradient_next, 2.0);

		baseline_spatial_scaling_backward(correct_gradient_input, correct_gradient_scales, gradient_next, input, scales);

//		channelScalingBackward(context, gradient_input, gradient_scales, gradient_next, input, scales);
//		EXPECT_LE(testing::diffForTest(correct_gradient_input, gradient_input), 1.0e-6f);
//		EXPECT_LE(testing::diffForTest(correct_gradient_scales, gradient_scales), 1.0e-6f);

		if (testing::has_device_supporting(DataType::FLOAT32))
		{
			const Device device = testing::get_device_for_test();
			Context context(device);
			input.moveTo(device);
			scales.moveTo(device);
			gradient_next.moveTo(device);
			gradient_input.moveTo(device);
			gradient_scales.moveTo(device);
			gradient_input.zeroall();
			gradient_scales.zeroall();
			spatialScalingBackward(context, 1.0f, gradient_next, input, scales, 0.0f, gradient_input, 0.0f, gradient_scales);
			context.synchronize();

			EXPECT_LE(testing::diffForTest(correct_gradient_input, gradient_input), 1.0e-5f);
			EXPECT_LE(testing::diffForTest(correct_gradient_scales, gradient_scales), 1.0e-5f);
		}
	}

} /* namespace ml */

