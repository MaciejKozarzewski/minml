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

namespace
{
	int get_backward_data_workspace_size(const ml::Shape &input_shape, int num_heads)
	{
		const int batch_size = input_shape[0];
		const int tokens = input_shape[1] * input_shape[2];
		return batch_size * num_heads * tokens * tokens;
	}
}

namespace ml
{
	MultiHeadAttention::MultiHeadAttention(int numberOfHeads, int positional_encoding_range) :
			Layer(),
			m_number_of_heads(numberOfHeads),
			m_positional_encoding_range(positional_encoding_range)
	{
	}

	Shape MultiHeadAttention::getWeightShape() const
	{
		const int tmp = 2 * m_positional_encoding_range - 1;
		return Shape( { m_number_of_heads, tmp, 4 * ((tmp + 3) / 4) });
	}
	void MultiHeadAttention::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer expects single input shape");
		if (shapes[0].rank() != 4)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer expects 4D input shape");
		if (shapes[0].lastDim() % 3 != 0)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer last dimension must be divisible by 3");
		if ((shapes[0].lastDim() / 3) % m_number_of_heads != 0)
			throw IllegalArgument(METHOD_NAME, "MultiHeadAttention layer last dimension must be divisible by number of heads");
		m_input_shapes = shapes;
	}
	Shape MultiHeadAttention::getOutputShape() const
	{
		const int batch_size = getInputShape().dim(0);
		const int height = getInputShape().dim(1);
		const int width = getInputShape().dim(2);
		const int embedding = getInputShape().dim(3) / 3;
		return Shape( { batch_size, height, width, embedding });
	}

	std::string MultiHeadAttention::name() const
	{
		return "MultiHeadAttention";
	}
	Json MultiHeadAttention::getConfig() const
	{
		Json result = Layer::getConfig();
		result["number_of_heads"] = m_number_of_heads;
		result["positional_encoding_range"] = m_positional_encoding_range;
		return result;
	}

	int MultiHeadAttention::getWorkspaceSize() const noexcept
	{
		return multiHeadAttentionGetWorkspaceSize(context(), getInputShape(), getWeightShape(), isTrainable());
	}
	std::unique_ptr<Layer> MultiHeadAttention::clone(const Json &config) const
	{
		std::unique_ptr<MultiHeadAttention> result = std::make_unique<MultiHeadAttention>(config["number_of_heads"].getInt(),
				config["positional_encoding_range"].getInt());
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	void MultiHeadAttention::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		const int tmp = get_backward_data_workspace_size(input[0].shape(), m_number_of_heads);
		if (isTrainable() and m_backward_data.volume() < tmp)
			m_backward_data = Tensor( { tmp }, dtype(), device());

		multiHeadAttentionForward(context(), input.at(0), output, getWeights().getParam(), *m_workspace.lock(), m_backward_data);
	}
	void MultiHeadAttention::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		assert(input.size() == 1);
		assert(gradient_prev.size() == 1);
		multiHeadAttentionBackward(context(), input.at(0), getWeights().getParam(), gradient_prev.at(0), gradient_next, getWeights().getGradient(),
				*m_workspace.lock(), m_backward_data);
	}

} /* namespace ml */

