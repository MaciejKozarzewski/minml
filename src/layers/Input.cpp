/*
 * InputLayer.cpp
 *
 *  Created on: Feb 22, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Input.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	Input::Input(Shape input_shape) :
			Layer()
	{
		m_input_shapes.push_back(input_shape);
	}

	void Input::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Input layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Input::getOutputShape() const
	{
		return m_input_shapes[0];
	}
	void Input::setupQuantization(const std::vector<AffineTransform> &input_transforms, const AffineTransform &output_transform, int bits)
	{
		m_input_transforms = input_transforms;
		m_output_transform = output_transform;
		if (isQuantizable())
		{
			getWeights().getParam() = Tensor(Shape(), get_quantized_dtype(bits), device());
			getBias().getParam() = zeros_like(getWeights().getParam());
		}
	}

	std::string Input::name() const
	{
		return "Input";
	}
	Json Input::getConfig() const
	{
		Json result = Layer::getConfig();
		result["input_shape"] = getInputShape().serialize();
		return result;
	}

	std::unique_ptr<Layer> Input::clone(const Json &config) const
	{
		std::unique_ptr<Input> result = std::make_unique<Input>(Shape(config["input_shape"]));
		result->m_activation = activationFromString(config["nonlinearity"]);
		result->loadConfig(config);
		return std::unique_ptr<Layer>(static_cast<Layer*>(result.release()));
	}

	void Input::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		activationForward(context(), 1.0f, output, 0.0f, output, m_activation);
	}
	void Input::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		activationBackward(context(), 1.0f, gradient_next, output, 0.0f, gradient_next, m_activation);
	}

} /* namespace ml */

