/*
 * Router.cpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Router.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	Router::Router(int experts) :
			Layer(),
			m_experts(experts)
	{
	}

	void Router::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "Router can onyl have 1 input");
		m_input_shapes = shapes;
	}
	Shape Router::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		const int batch_size = getInputShape().firstDim();
		int tokens = 1;
		for (int i = 1; i < getInputShape().rank() - 1; i++)
			tokens *= getInputShape().dim(i);
		return Shape( { batch_size, m_experts, tokens });
	}
	Shape Router::getWeightShape() const
	{
		return Shape( { m_experts, getInputShape().lastDim() });
	}

	std::string Router::name() const
	{
		return "Router";
	}
	Json Router::getConfig() const
	{
		Json result = Layer::getConfig();
		result["experts"] = m_experts;
		return result;
	}

	std::unique_ptr<Layer> Router::clone(const Json &config) const
	{
		std::unique_ptr<Router> result = std::make_unique<Router>(config["experts"]);
		result->loadConfig(config);
		return result;
	}
	int Router::getWorkspaceSize() const noexcept
	{
		return getInputShape().firstDim() * m_experts * getInputShape().lastDim();
	}
	void Router::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const Tensor flattened_input = input[0].view().flatten( { 1, 2 });
		gemmBatched(context(), 'n', 't', output, getWeights().getParam(), flattened_input, 1.0f, 0.0f);
		Tensor y = output.view( { output.dim(0) * output.dim(1), output.dim(2) });
		softmaxForward(context(), y, y);
	}
	void Router::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		const Tensor flattened_y = output.view().flatten( { 0, 1 });
		Tensor flattened_dy = gradient_next.view().flatten( { 0, 1 });
		softmaxBackward(context(), 1.0f, flattened_dy, flattened_y, 0.0f, flattened_dy);

		const Tensor flattened_x = input[0].view().flatten( { 1, 2 });
		Tensor flattened_dx = gradient_prev[0].view().flatten( { 1, 2 });
		gemmBatched(context(), 't', 'n', flattened_dx, gradient_next, getWeights().getParam(), 1.0f, beta[0]);

		Tensor tmp_dw = m_workspace.lock()->view( { input[0].firstDim(), m_experts, input[0].lastDim() });
		gemmBatched(context(), 'n', 'n', tmp_dw, gradient_next, flattened_dx, 1.0f, 0.0f);

		Tensor flattened_dw = getWeights().getGradient().view().flatten();
		Tensor flattened_tmp_dw = tmp_dw.view().flatten( { 1, 2 });
		sumOverFirstDim(context(), 1.0f, flattened_dw, 0.0f, flattened_tmp_dw);
	}

} /* namespace ml */

