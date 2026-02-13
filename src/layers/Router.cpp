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
#include <minml/utils/testing_util.hpp>

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
		return Shape( { getInputShape().dim(0), m_experts, getInputShape().dim(1), getInputShape().dim(2) });
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
		return getInputShape().firstDim() * getWeightShape().volume();
	}
	void Router::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const Tensor flattened_input = input[0].view().flatten( { 1, 2 });
		Tensor flattened_output = output.view().flatten( { 2, 3 });
		gemmBatched(context(), 'n', 't', flattened_output, getWeights().getParam(), flattened_input, 1.0f, 0.0f);
		Tensor y = output.view().flatten( { 0, 1 }, { 2, 3 });
		softmaxForward(context(), y, y);
	}
	void Router::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);

		{ /* artificial scope for variables */
			const Tensor flattened_y = output.view().flatten( { 0, 1 }, { 2, 3 }); // (NE) x (HW)
			Tensor flattened_dy = gradient_next.view().flatten( { 0, 1 }, { 2, 3 }); // (NE) x (HW)
			softmaxBackward(context(), 1.0f, flattened_dy, flattened_y, 0.0f, flattened_dy);
		}

		const Tensor flattened_x = input[0].view().flatten( { 1, 2 }); // N x (HW) x C
		Tensor flattened_dx = gradient_prev[0].view().flatten( { 1, 2 }); // N x (HW) x C
		const Tensor flattened_y = output.view().flatten( { 2, 3 }); // N x E x (HW)
		Tensor flattened_dy = gradient_next.view().flatten( { 2, 3 }); // N x E x (HW)
		gemmBatched(context(), 't', 'n', flattened_dx, flattened_dy, getWeights().getParam(), 1.0f, beta[0]);

		Tensor tmp_dw = m_workspace.lock()->view( { input[0].firstDim(), m_experts, input[0].lastDim() });
		gemmBatched(context(), 'n', 'n', tmp_dw, flattened_dy, flattened_x, 1.0f, 0.0f);

		Tensor flattened_dw = getWeights().getGradient().view().flatten(); // (EC)
		Tensor flattened_tmp_dw = tmp_dw.view().flatten( { 1, 2 }); // N x (EC)
		sumOverFirstDim(context(), 1.0f, flattened_tmp_dw, 0.0f, flattened_dw);
	}

} /* namespace ml */

