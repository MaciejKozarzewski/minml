/*
 * MatMul.cpp
 *
 *  Created on: Feb 13, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/MatMul.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace
{
	ml::Tensor flatten_input_tensor(const ml::Tensor &t)
	{
		const int first_dim = t.firstDim();
		const int last_dim = t.shape().volumeWithoutFirstDim();
		return t.view(ml::Shape( { first_dim, last_dim }));
	}
}

namespace ml
{
	MatMul::MatMul(int rows, int columns, char inputMode, bool useBias) :
			Layer(),
			m_rows(rows),
			m_columns(columns),
			m_input_mode(inputMode),
			m_use_bias(useBias)
	{
		assert(inputMode == 'n' || inputMode == 't');
	}

	void MatMul::setInputShape(const std::vector<Shape> &shapes)
	{
		if (m_input_shapes.size() != 1)
			throw LogicError(METHOD_NAME, "MatMul expects exactly one input");
		m_input_shapes = shapes;
	}
	Shape MatMul::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
//		return Shape( { getInputShape().firstDim(), m_neurons });
	}
	Shape MatMul::getWeightShape() const
	{
		return Shape( { m_rows, m_columns });
	}
	Shape MatMul::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_columns });
		else
			return Shape();
	}

	std::string MatMul::name() const
	{
		return "MatMul";
	}
	Json MatMul::getConfig() const
	{
		Json result = Layer::getConfig();
		result["rows"] = m_rows;
		result["columns"] = m_columns;
		result["input_mode"] = m_input_mode;
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> MatMul::clone(const Json &config) const
	{
		std::unique_ptr<MatMul> result = std::make_unique<MatMul>(config["rows"].getInt(), config["columns"].getInt(), config["input_mode"].getChar(),
				config["use_bias"].getBool());
		result->loadConfig(config);
		return result;
	}

	void MatMul::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		if (input[0].rank() == 2)
		{
			assert(output.rank() == 2);
			gemm_ex(context(), output, 1.0f, m_input_mode, input[0], 'n', getWeights().getParam(), 0.0f, output, getBias().getParam(), m_activation);
		}
		if (input[0].rank() == 3)
		{
			gemmBatched(context(), m_input_mode, 'n', output, input[0], getWeights().getParam(), 1.0f, 0.0f);
			if (m_use_bias)
			{
				const Tensor flattened_input = flatten_input_tensor(input[0]);
//				addBiasAct(context(), output, flattened_input, getBias().getParam(), m_activation);
			}
		}
	}
	void MatMul::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

//		activationBackward(context(), gradient_next, gradient_next, output, m_activation);
//		if (isUsingWeights())
//		{
//			Tensor tmp_grad = flatten_input_tensor(gradient_prev[0]);
//			gemm(context(), 'n', 'n', tmp_grad, gradient_next, getWeights().getParam(), 1, 0);
//			gemm(context(), 't', 'n', getWeights().getGradient(), gradient_next, flatten_input_tensor(input[0]), 1, 0);
//		}
//		else
//		{
//			Tensor tmp = flatten_input_tensor(gradient_prev[0]);
//			tmp.copyFrom(context(), gradient_next);
//		}
//
//		if (isUsingBias())
//			sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 0);
	}

} /* namespace ml */

