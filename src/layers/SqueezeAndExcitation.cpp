/*
 * SqueezeAndExcitation.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/SqueezeAndExcitation.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/utils/json.hpp>

#include <minml/utils/time_util.hpp>

namespace ml
{
	SqueezeAndExcitation::SqueezeAndExcitation() :
			Layer()
	{
	}
	SqueezeAndExcitation::SqueezeAndExcitation(const Dense &dense1, const Dense &dense2) :
			Layer(),
			m_dense_1_weights(dense1.getWeights().getParam()),
			m_dense_1_bias(dense1.getBias().getParam()),
			m_dense_2_weights(dense2.getWeights().getParam()),
			m_dense_2_bias(dense2.getBias().getParam())
	{
	}

	Shape SqueezeAndExcitation::getOutputShape() const
	{
		return getInputShape();
	}

	std::string SqueezeAndExcitation::name() const
	{
		return "SqueezeAndExcitation";
	}
	Json SqueezeAndExcitation::getConfig() const
	{
		return Layer::getConfig();
	}
	Json SqueezeAndExcitation::saveParameters(SerializedObject &binary_data) const
	{
		Json result = Layer::saveParameters(binary_data);
		result["dense_1_weights"] = m_dense_1_weights.serialize(binary_data);
		result["dense_1_bias"] = m_dense_1_bias.serialize(binary_data);
		result["dense_2_weights"] = m_dense_2_weights.serialize(binary_data);
		result["dense_2_bias"] = m_dense_2_bias.serialize(binary_data);
		return result;
	}
	void SqueezeAndExcitation::loadParameters(const Json &json, const SerializedObject &binary_data)
	{
		m_dense_1_weights.unserialize(json["dense_1_weights"], binary_data);
		m_dense_1_bias.unserialize(json["dense_1_bias"], binary_data);
		m_dense_2_weights.unserialize(json["dense_2_weights"], binary_data);
		m_dense_2_bias.unserialize(json["dense_2_bias"], binary_data);
	}

	std::unique_ptr<Layer> SqueezeAndExcitation::clone(const Json &config) const
	{
		std::unique_ptr<SqueezeAndExcitation> result = std::make_unique<SqueezeAndExcitation>();
		result->loadConfig(config);
		return result;
	}

	int SqueezeAndExcitation::getWorkspaceSize() const noexcept
	{
		const int batch_size = getInputShape().firstDim();
		const int internal_channels_0 = getInputShape().lastDim();
		const int internal_channels_1 = m_dense_1_weights.firstDim();
		return batch_size * (internal_channels_0 + internal_channels_1);
	}
	void SqueezeAndExcitation::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		m_dense_1_weights.moveTo(device());
		m_dense_1_bias.moveTo(device());
		m_dense_2_weights.moveTo(device());
		m_dense_2_bias.moveTo(device());
	}
	void SqueezeAndExcitation::convertTo(DataType newType)
	{
		Layer::convertTo(newType);
		m_dense_1_weights.convertTo(context(), newType);
		m_dense_1_bias.convertTo(context(), newType);
		m_dense_2_weights.convertTo(context(), newType);
		m_dense_2_bias.convertTo(context(), newType);
	}

	void SqueezeAndExcitation::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);

		const int batch_size = input[0].firstDim();
		const int channels = input[0].lastDim();
		const int internal_channels = m_dense_1_weights.firstDim();

		Tensor tmp0 = m_workspace.lock()->view( { batch_size, channels });
		Tensor tmp1 = m_workspace.lock()->view( { batch_size, internal_channels }, tmp0.volume());

		globalAveragePoolingForward(context(), 1.0f, input[0], 0.0f, tmp0);
		gemm_ex(context(), tmp1, 1, 'n', tmp0, 't', m_dense_1_weights, 0, tmp1, m_dense_1_bias, ActivationType::RELU);
		gemm_ex(context(), tmp0, 1, 'n', tmp1, 't', m_dense_2_weights, 0, tmp0, m_dense_2_bias, ActivationType::SIGMOID);
		channelScalingForward(context(), 1.0f, input[0], tmp0, 0.0f, output);
	}
	void SqueezeAndExcitation::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next, const std::vector<float> &beta)
	{
		throw LogicError(METHOD_NAME, "SqueezeAndExcitation is not a trainable layer");
	}

} /* namespace ml */

