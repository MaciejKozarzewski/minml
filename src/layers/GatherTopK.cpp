/*
 * GatherTopK.cpp
 *
 *  Created on: Feb 8, 2026
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/GatherTopK.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Context.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

namespace ml
{
	GatherTopK::GatherTopK(int top_k) :
			Layer(),
			m_top_k(top_k)
	{
	}
	void GatherTopK::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 2)
			throw LogicError(METHOD_NAME, "gather top K must have 2 inputs");
		const int batch_size = shapes[0].dim(0);
		const int height = shapes[0].dim(1);
		const int width = shapes[0].dim(2);
		const int channels = shapes[0].dim(3);

		if (batch_size != shapes[1].dim(0))
			throw LogicError(METHOD_NAME, "batch size mismatch");
		const int experts = shapes[1].dim(1);
		if (height != shapes[1].dim(2))
			throw LogicError(METHOD_NAME, "height mismatch");
		if (width != shapes[1].dim(3))
			throw LogicError(METHOD_NAME, "width mismatch");

		m_indices_cache = Tensor( { batch_size, experts, m_top_k }, "int32", device());
		m_input_shapes = shapes;
	}
	Shape GatherTopK::getOutputShape() const
	{
		const int batch_size = getInputShape(0).firstDim();
		const int channels = getInputShape(0).lastDim();
		const int experts = getInputShape(1).dim(1);
		return Shape( { batch_size, m_top_k, experts, channels });
	}
	std::string GatherTopK::name() const
	{
		return "GatherTopK";
	}
	Json GatherTopK::getConfig() const
	{
		Json result = Layer::getConfig();
		result["top_k"] = m_top_k;
		return result;
	}
	std::unique_ptr<Layer> GatherTopK::clone(const Json &config) const
	{
		std::unique_ptr<GatherTopK> result = std::make_unique<GatherTopK>(config["top_k"].getInt());
		result->loadConfig(config);
		return result;
	}
	void GatherTopK::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		m_indices_cache.moveTo(context->device());
	}
	void GatherTopK::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 2);
		Tensor values;

		const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });

		selectTopK(context(), flattened_router_output, m_indices_cache, values);
		gatherTokensForward(context(), input[0], m_indices_cache, 0.0f, output);
	}
	void GatherTopK::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		assert(input.size() == 2);
		assert(gradient_prev.size() == 2);

		const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });
		gatherTokensBackward(context(), gradient_next, m_indices_cache, beta[0], gradient_prev[0]);
	}

} /* namespace ml */

