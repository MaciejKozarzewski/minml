/*
 * ScatterTopK.cpp
 *
 *  Created on: Feb 10, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/ScatterTopK.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Context.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	ScatterTopK::ScatterTopK() :
			Layer()
	{
	}
	void ScatterTopK::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 2)
			throw LogicError(METHOD_NAME, "gather top K must have 2 inputs");
		const int experts = shapes[0].dim(0);
		const int batch_size = shapes[0].dim(1);
		const int top_k = shapes[0].dim(2);
		const int channels = shapes[0].dim(3);

		if (batch_size != shapes[1].dim(0))
			throw LogicError(METHOD_NAME, "batch size mismatch");
		if (experts != shapes[1].dim(1))
			throw LogicError(METHOD_NAME, "number of experts mismatch");
		const int height = shapes[1].dim(2);
		const int width = shapes[1].dim(3);

		m_indices_cache = Tensor( { batch_size, experts, top_k }, "int32", device());
		m_values_cache = Tensor( { batch_size, experts, top_k }, dtype(), device());

		m_input_shapes = shapes;
	}
	Shape ScatterTopK::getOutputShape() const
	{
		const int batch_size = getInputShape(0).dim(1);
		const int height = getInputShape(1).dim(2);
		const int width = getInputShape(1).dim(3);
		const int channels = getInputShape(0).lastDim();
		return Shape( { batch_size, height, width, channels });
	}
	std::string ScatterTopK::name() const
	{
		return "ScatterTopK";
	}
	Json ScatterTopK::getConfig() const
	{
		Json result = Layer::getConfig();
		return result;
	}
	std::unique_ptr<Layer> ScatterTopK::clone(const Json &config) const
	{
		std::unique_ptr<ScatterTopK> result = std::make_unique<ScatterTopK>();
		result->loadConfig(config);
		return result;
	}
	void ScatterTopK::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		m_indices_cache.moveTo(context->device());
		m_values_cache.moveTo(context->device());
	}
	void ScatterTopK::convertTo(DataType newType)
	{
		Layer::convertTo(newType);
		m_values_cache.convertTo(context(), newType);
	}
	void ScatterTopK::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 2);

		const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });

		selectTopK(context(), flattened_router_output, m_indices_cache, m_values_cache);
		scatterTokensForward(context(), input[0], m_indices_cache, m_values_cache, 0.0f, output);
	}
	void ScatterTopK::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 2);
		assert(gradient_prev.size() == 2);
		scatterTokensBackward(context(), gradient_next, input[0], m_indices_cache, m_values_cache, beta[0], gradient_prev[0], beta[1],
				gradient_prev[1]);
	}

} /* namespace ml */
