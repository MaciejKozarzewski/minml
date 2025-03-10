/*
 * Layer.cpp
 *
 *  Created on: Nov 10, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Layer.hpp>
#include <minml/layers/quantization.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Device.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/testing_util.hpp>

#include <minml/training/Optimizer.hpp>
#include <minml/training/Regularizer.hpp>
#include <minml/training/Initializer.hpp>

#include <minml/layers/ChannelScaling.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/layers/DepthToSpace.hpp>
#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/layers/FusedConvBlock.hpp>
#include <minml/layers/Input.hpp>
#include <minml/layers/Add.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/LayerNormalization.hpp>
#include <minml/layers/LearnableGlobalPooling.hpp>
#include <minml/layers/Gelu.hpp>
#include <minml/layers/GlobalAveragePooling.hpp>
#include <minml/layers/GlobalBroadcastHW.hpp>
#include <minml/layers/GlobalPooling.hpp>
#include <minml/layers/MultiHeadAttention.hpp>
#include <minml/layers/PositionalEncoding.hpp>
#include <minml/layers/RMSNormalization.hpp>
#include <minml/layers/Softmax.hpp>
#include <minml/layers/SpaceToDepth.hpp>
#include <minml/layers/SqueezeAndExcitation.hpp>
#include <minml/layers/WindowPartitioning.hpp>
#include <minml/layers/WindowMerging.hpp>
#include <unordered_map>
#include <cmath>
#include <mutex>

namespace
{
	using namespace ml;

	Json to_json(const AffineTransform &tq)
	{
		return Json( { { "scale", tq.scale() }, { "shift", tq.shift() } });
	}
	AffineTransform tensor_quantizer_from_json(const Json &json)
	{
		return AffineTransform(json["scale"].getDouble(), json["shift"].getDouble());
	}
}

namespace ml
{
	std::string toString(ActivationType act)
	{
		switch (act)
		{
			default:
			case ActivationType::LINEAR:
				return "linear";
			case ActivationType::SIGMOID:
				return "sigmoid";
			case ActivationType::TANH:
				return "tanh";
			case ActivationType::RELU:
				return "relu";
			case ActivationType::GELU:
				return "gelu";
			case ActivationType::EXP:
				return "exp";
			case ActivationType::SOFTMAX:
				return "softmax";
		}
	}
	ActivationType activationFromString(const std::string &str)
	{
		if (str == "linear")
			return ActivationType::LINEAR;
		if (str == "sigmoid")
			return ActivationType::SIGMOID;
		if (str == "tanh")
			return ActivationType::TANH;
		if (str == "relu")
			return ActivationType::RELU;
		if (str == "gelu")
			return ActivationType::GELU;
		if (str == "exp")
			return ActivationType::EXP;
		if (str == "softmax")
			return ActivationType::SOFTMAX;
		throw LogicError(METHOD_NAME, "unknown nonlinearity '" + str + "'");
	}

	Layer::Layer(std::string activation, DataType dtype) :
			m_dtype(dtype),
			m_activation(activationFromString(activation))
	{
	}

	bool Layer::isTrainable() const noexcept
	{
		return getWeights().isTrainable() or getBias().isTrainable();
	}
	bool Layer::isQuantizable() const noexcept
	{
		return m_is_quantizable;
	}

	ActivationType Layer::getActivationType() const noexcept
	{
		return m_activation;
	}
	void Layer::setActivationType(ActivationType act) noexcept
	{
		m_activation = act;
	}
	Layer& Layer::quantizable(bool b) noexcept
	{
		m_is_quantizable = b;
		return *this;
	}
	void Layer::setupQuantization(const std::vector<AffineTransform> &input_transforms, const AffineTransform &output_transform, int bits)
	{
		m_input_transforms = input_transforms;
		m_output_transform = output_transform;
		if (isQuantizable())
		{
			const std::string mode = (name() == "DepthwiseConv2D") ? "per_last_dim" : "per_first_dim";
			const WeightQuantizer tmp = quantize_weights(getWeights().getParam(), getBias().getParam(), input_transforms.at(0), mode, bits);
			m_channel_scales = tmp.channel_scales;
			getWeights().getParam() = tmp.weights;
			getBias().getParam() = tmp.bias;
			m_quantization_bits = bits;
		}
	}

	Json Layer::getConfig() const
	{
		Json result;
		result["name"] = name();
		result["nonlinearity"] = toString(m_activation);
		result["dtype"] = toString(m_dtype);
		result["is_quantizable"] = m_is_quantizable;
		result["input_transforms"] = Json::array();
		for (size_t i = 0; i < m_input_transforms.size(); i++)
			result["input_transforms"][i] = to_json(m_input_transforms[i]);
		result["output_transform"] = to_json(m_output_transform);
		return result;
	}
	void Layer::loadConfig(const Json &config)
	{
		this->m_dtype = typeFromString(config["dtype"].getString());
		if (config.hasKey("is_quantizable"))
			this->m_is_quantizable = config["is_quantizable"].getBool();
		if (config.hasKey("input_transforms"))
		{
			for (int i = 0; i < config["input_transforms"].size(); i++)
				m_input_transforms.push_back(tensor_quantizer_from_json(config["input_transforms"][i]));
		}
		if (config.hasKey("output_transform"))
			m_output_transform = tensor_quantizer_from_json(config["output_transform"]);
	}
	Json Layer::saveParameters(SerializedObject &binary_data) const
	{
		Json result;
		result["weights"] = (m_weights == nullptr) ? Json::null() : m_weights->serialize(binary_data);
		result["bias"] = (m_bias == nullptr) ? Json::null() : m_bias->serialize(binary_data);
		result["channel_scales"] = m_channel_scales.isEmpty() ? Json::null() : m_channel_scales.serialize(binary_data);
		return result;
	}
	void Layer::loadParameters(const Json &json, const SerializedObject &binary_data)
	{
		if (json.hasKey("weights") and not json["weights"].isNull())
			getWeights().unserialize(json["weights"], binary_data);
		if (json.hasKey("bias") and not json["bias"].isNull())
			getBias().unserialize(json["bias"], binary_data);
		if (json.hasKey("channel_scales") and not json["channel_scales"].isNull())
			m_channel_scales.unserialize(json["channel_scales"], binary_data);
	}

	int Layer::numberOfInputs() const noexcept
	{
		return static_cast<int>(m_input_shapes.size());
	}
	void Layer::setInputShape(const Shape &shape)
	{
		m_input_shapes = { shape };
	}
	void Layer::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	const std::vector<Shape>& Layer::getInputShapes() const noexcept
	{
		return m_input_shapes;
	}
	Shape Layer::getInputShape(int index) const
	{
		if (index < 0 || index >= static_cast<int>(m_input_shapes.size()))
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_input_shapes.size());
		return m_input_shapes[index];
	}
	Shape Layer::getWeightShape() const
	{
		return Shape();
	}
	Shape Layer::getBiasShape() const
	{
		return Shape();
	}

	Device Layer::device() const
	{
		if (m_context.lock() == nullptr)
			return Device::cpu();
		else
			return context().device();
	}
	DataType Layer::dtype() const noexcept
	{
		return m_dtype;
	}
	const Context& Layer::context() const
	{
		if (m_context.lock() == nullptr)
			throw UninitializedObject(METHOD_NAME, "context was not initialized");
		else
			return *m_context.lock();
	}

	Parameter& Layer::getWeights()
	{
		if (m_weights == nullptr)
			m_weights = std::make_unique<Parameter>(getWeightShape(), dtype(), device());
		return *m_weights;
	}
	Parameter& Layer::getBias()
	{
		if (m_bias == nullptr)
			m_bias = std::make_unique<Parameter>(getBiasShape(), dtype(), device());
		return *m_bias;
	}
	const Parameter& Layer::getWeights() const
	{
		return *m_weights;
	}
	const Parameter& Layer::getBias() const
	{
		return *m_bias;
	}

	void Layer::convertTo(DataType newType)
	{
		if (isInteger(newType) == false and isInteger(dtype()) == false)
		{
			getWeights().getParam().convertTo(context(), newType);
			getBias().getParam().convertTo(context(), newType);
		}
		m_dtype = newType;
	}

	int Layer::getWorkspaceSize() const noexcept
	{
		return 0;
	}
	void Layer::setWorkspace(std::shared_ptr<Tensor> &workspace)
	{
		assert(workspace->device() == device());
		m_workspace = workspace;
	}

	void Layer::changeContext(std::shared_ptr<Context> &context)
	{
		this->m_context = context;
		m_channel_scales.moveTo(device());
		if (m_weights != nullptr)
			getWeights().moveTo(device());
		if (m_bias != nullptr)
			getBias().moveTo(device());
	}

	void Layer::init()
	{
		m_initializer.init_weights(context(), getWeights(), 1.0f / std::sqrt(getWeightShape().volumeWithoutFirstDim()), 0.0f);
		m_initializer.init_bias(context(), getBias(), 0.1f, 0.0f);
	}
	void Layer::setOptimizer(const Optimizer &optimizer)
	{
		getWeights().getOptimizer() = optimizer;
		getBias().getOptimizer() = optimizer;
	}
	void Layer::setRegularizer(const Regularizer &regularizer)
	{
		getWeights().getRegularizer() = regularizer;
		getBias().getRegularizer() = regularizer;
	}

	void Layer::learn()
	{
		getWeights().learn(context());
		getBias().learn(context());
	}

	std::unique_ptr<Layer> loadLayer(const Json &json, const SerializedObject &binary_data)
	{
		static const Add add;
		static const BatchNormalization batchnorm;
		static const ChannelScaling channel_scaling;
		static const Conv2D conv2d(0, 0);
		static const Dense dense(0);
		static const DepthToSpace depth_to_space(0, { 0, 0 });
		static const DepthwiseConv2D depthwise_conv2d(0, 0);
		static const FusedConvBlock fused_conv_block;
		static const Gelu gelu;
		static const GlobalAveragePooling global_average_pooling;
		static const GlobalBroadcastHW global_broadcast;
		static const GlobalPooling global_pooling;
		static const LayerNormalization layernorm;
		static const LearnableGlobalPooling learnable_global_pooling(0);
		static const MultiHeadAttention mha(0, 0, false);
		static const Input input;
		static const RMSNormalization rmsnorm;
		static const PositionalEncoding positional_encoding;
		static const Softmax softmax( { 0 });
		static const SpaceToDepth space_to_depth(0);
		static const SqueezeAndExcitation se;
		static const WindowPartitioning window_partitioning(0, 0);
		static const WindowMerging window_merging(Shape(), 0);

		const std::string name = json["name"];
		std::unique_ptr<Layer> result;

		if (name == add.name())
			result = add.clone(json);
		if (name == batchnorm.name())
			result = batchnorm.clone(json);
		if (name == channel_scaling.name())
			result = channel_scaling.clone(json);
		if (name == conv2d.name())
			result = conv2d.clone(json);
		if (name == dense.name())
			result = dense.clone(json);
		if (name == depth_to_space.name())
			result = depth_to_space.clone(json);
		if (name == depthwise_conv2d.name())
			result = depthwise_conv2d.clone(json);
		if (name == fused_conv_block.name())
			result = fused_conv_block.clone(json);
		if (name == gelu.name())
			result = gelu.clone(json);
		if (name == global_average_pooling.name())
			result = global_average_pooling.clone(json);
		if (name == global_broadcast.name())
			result = global_broadcast.clone(json);
		if (name == global_pooling.name())
			result = global_pooling.clone(json);
		if (name == layernorm.name())
			result = layernorm.clone(json);
		if (name == learnable_global_pooling.name())
			result = learnable_global_pooling.clone(json);
		if (name == mha.name())
			result = mha.clone(json);
		if (name == input.name())
			result = input.clone(json);
		if (name == positional_encoding.name())
			result = positional_encoding.clone(json);
		if (name == rmsnorm.name())
			result = rmsnorm.clone(json);
		if (name == softmax.name())
			result = softmax.clone(json);
		if (name == space_to_depth.name())
			result = space_to_depth.clone(json);
		if (name == se.name())
			result = se.clone(json);
		if (name == window_partitioning.name())
			result = window_partitioning.clone(json);
		if (name == window_merging.name())
			result = window_merging.clone(json);

		if (result == nullptr)
			throw LogicError(METHOD_NAME, "unknown layer '" + static_cast<std::string>(json["name"]) + "'");
		return result;
	}

} /* namespace ml */

