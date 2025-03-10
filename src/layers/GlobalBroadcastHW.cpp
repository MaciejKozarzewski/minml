/*
 * GlobalBroadcastHW.cpp
 *
 *  Created on: Feb 16, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/GlobalBroadcastHW.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

#include <minml/utils/time_util.hpp>

namespace ml
{
	GlobalBroadcastHW::GlobalBroadcastHW(std::string activation, bool use_bias) :
			Layer(activation)
	{
		m_use_bias = use_bias;
	}
	GlobalBroadcastHW& GlobalBroadcastHW::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool GlobalBroadcastHW::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void GlobalBroadcastHW::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "GlobalBroadcastHW layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape GlobalBroadcastHW::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return getInputShape();
	}
	Shape GlobalBroadcastHW::getWeightShape() const
	{
		const int hw = getInputShape()[1] * getInputShape()[2];
		return Shape( { hw, hw });
	}
	Shape GlobalBroadcastHW::getBiasShape() const
	{
		if (isUsingBias())
			return Shape( { getInputShape()[1] * getInputShape()[2] });
		else
			return Shape();
	}

	Json GlobalBroadcastHW::getConfig() const
	{
		Json result = Layer::getConfig();
		result["use_bias"] = m_use_bias;
		return result;
	}
	std::string GlobalBroadcastHW::name() const
	{
		return "GlobalBroadcastHW";
	}

	int GlobalBroadcastHW::getWorkspaceSize() const noexcept
	{
		return getWeightShape().volume() + 2 * getInputShape().volume();
	}
	std::unique_ptr<Layer> GlobalBroadcastHW::clone(const Json &config) const
	{
		std::unique_ptr<GlobalBroadcastHW> result = std::make_unique<GlobalBroadcastHW>(config["nonlinearity"], config["use_bias"]);
		result->loadConfig(config);
		return result;
	}

	void GlobalBroadcastHW::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		struct Timer
		{
				std::string m_name;
				double m_start = 0.0;
				double m_total_time = 0.0;
				int m_count = 0;
				bool m_init = false;

				Timer(const std::string &name) :
						m_name(name)
				{
				}
				~Timer()
				{
					std::cout << m_name << " : " << 1.0e3 * m_total_time / m_count << " ms\n";
				}
				void start() noexcept
				{
					m_start = getTime();
				}
				void stop() noexcept
				{
					if (m_init)
					{
						m_total_time += getTime() - m_start;
						m_count++;
					}
					else
						m_init = true;
				}

		};
		static Timer transpose("transpose  ");
		static Timer copying("copying    ");
		static Timer matrix_multiply("gemm       ");
		static Timer nonlinearity("activation ");

		assert(input.size() == 1);
		const bool emulate_low_precision = false; //isTrainable() and dtype() == DataType::FLOAT32;

		Tensor tmp_w;
//		if (emulate_low_precision)
//		{
//			tmp_w = m_workspace.lock()->view(getWeightShape());
//			emulateLowPrecision(context(), tmp_w, getWeights().getParam());
//		}
//		else
			tmp_w = getWeights().getParam().view();

		const int batch_size = input[0].dim(0);
		const int hw = input[0].dim(1) * input[0].dim(2);
		const int channels = input[0].dim(3);

		const Shape tmp_shape( { batch_size, channels, hw });
		Tensor tmp_in = m_workspace.lock()->view(tmp_shape, tmp_w.volume());
		Tensor tmp_out = m_workspace.lock()->view(tmp_shape, tmp_w.volume() + tmp_in.volume());

		Tensor input_view = input[0].view( { batch_size, hw, channels });
		transpose.start();
		transpose_021(context(), input_view, tmp_in);
		transpose.stop();

		copying.start();
		tmp_out.copyFrom(context(), tmp_in); // skip connection from input to output
		copying.stop();

		tmp_in.reshape( { batch_size * channels, hw });
		tmp_out.reshape( { batch_size * channels, hw });
		matrix_multiply.start();
		gemm(context(), 'n', 't', tmp_out, tmp_in, tmp_w, 1.0f, 1.0f);
		matrix_multiply.stop();

		tmp_in.reshape(tmp_shape);
		tmp_out.reshape(tmp_shape);
		Tensor output_view = output.view( { batch_size, hw, channels });
		transpose.start();
		transpose_021(context(), tmp_out, output_view);
		transpose.stop();

		nonlinearity.start();
		if (isUsingBias())
			addBiasAct(context(), output, output, getBias().getParam(), m_activation);
		else
			activationForward(context(), output, output, m_activation);
		nonlinearity.stop();
	}
	void GlobalBroadcastHW::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		assert(input.size() == 1);
		const bool emulate_low_precision = false; //isTrainable() and dtype() == DataType::FLOAT32;

		activationBackward(context(), gradient_next, gradient_next, output, m_activation);
		if (isUsingBias())
			sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 0.0f);

		Tensor tmp_w;
//		if (emulate_low_precision)
//		{
//			tmp_w = m_workspace.lock()->view(getWeightShape());
//			emulateLowPrecision(context(), tmp_w, getWeights().getParam());
//		}
//		else
			tmp_w = getWeights().getParam().view();

		const int batch_size = input[0].dim(0);
		const int hw = input[0].dim(1) * input[0].dim(2);
		const int channels = input[0].dim(3);
		const Shape tmp_shape( { batch_size, channels, hw });

		Tensor tmp_in = m_workspace.lock()->view(tmp_shape, tmp_w.volume());
		Tensor tmp_out = m_workspace.lock()->view(tmp_shape, tmp_w.volume() + tmp_in.volume());

		Tensor gradient_next_view = gradient_next.view( { batch_size, hw, channels });
		transpose_021(context(), gradient_next_view, tmp_out);
		tmp_in.reshape( { batch_size * channels, hw });
		tmp_out.reshape( { batch_size * channels, hw });
		gemm(context(), 'n', 'n', tmp_in, tmp_out, tmp_w, 1.0f, 0.0f);

		tmp_in.reshape(tmp_shape);
		Tensor gradient_prev_view = gradient_prev[0].view( { batch_size, hw, channels });
		transpose_021(context(), tmp_in, gradient_prev_view);

		Tensor input_view = input[0].view( { batch_size, hw, channels });
		transpose_021(context(), input_view, tmp_in);

		tmp_in.reshape( { batch_size * channels, hw });
		tmp_out.reshape( { batch_size * channels, hw });
		gemm(context(), 't', 'n', getWeights().getGradient(), tmp_out, tmp_in, 1.0f, 0.0f);

		addTensors(context(), gradient_prev[0], gradient_prev[0], gradient_next);
	}

} /* namespace ml */

