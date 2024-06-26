/*
 * MultiHeadAttention.cpp
 *
 *  Created on: Jun 12, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/MultiHeadAttention.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	MultiHeadAttention::MultiHeadAttention(int numberOfHeads) :
			Layer(),
			m_number_of_heads(numberOfHeads)
	{
	}

	void MultiHeadAttention::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer expects single input shape");
		if (shapes[0].rank() != 3)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer expects 3D input shape");
		if (shapes[0].lastDim() % 3 != 0)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer last dimension must be divisible by 3");
		if ((shapes[0].lastDim() / 3) % m_number_of_heads != 0)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer last dimension must be divisible by number of heads");
		m_input_shapes = shapes;
	}
	Shape MultiHeadAttention::getOutputShape() const
	{
		const int batch_size = getInputShape().dim(0);
		const int tokens = getInputShape().dim(1);
		const int embedding = getInputShape().dim(2) / 3;
		return Shape( { batch_size, tokens, embedding });
	}

	std::string MultiHeadAttention::name() const
	{
		return "MultiHeadAttention";
	}
	Json MultiHeadAttention::getConfig() const
	{
		Json result = Layer::getConfig();
		result["numberf_of_heads"] = m_number_of_heads;
		return result;
	}

	int MultiHeadAttention::getWorkspaceSize() const noexcept
	{
		return multiHeadAttentionGetWorkspaceSize(context(), getInputShape(), m_number_of_heads, isTrainable());
	}
	std::unique_ptr<Layer> MultiHeadAttention::clone(const Json &config) const
	{
		std::unique_ptr<MultiHeadAttention> result = std::make_unique<MultiHeadAttention>(config["number_of_heads"].getInt());
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	void MultiHeadAttention::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		multiHeadAttentionForward(context(), input.at(0), output, m_number_of_heads, *m_workspace.lock());
	}
	void MultiHeadAttention::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);
		multiHeadAttentionBackward(context(), input.at(0), gradient_prev.at(0), gradient_next, m_number_of_heads, *m_workspace.lock());
	}

} /* namespace ml */

