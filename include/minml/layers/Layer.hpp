/*
 * Layer.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_LAYER_HPP_
#define MINML_LAYERS_LAYER_HPP_

#include <minml/core/DataType.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/layers/quantization.hpp>
#include <minml/training/Initializer.hpp>

#include <memory>
#include <vector>

class Json;
class SerializedObject;
namespace ml /* forward declarations */
{
	class Context;
	class Shape;
	class Device;
	class Tensor;
}

namespace ml
{

	enum class ActivationType
	{
		LINEAR,
		SIGMOID,
		TANH,
		RELU,
		LEAKY_RELU,
		EXP,
		SOFTMAX
	};
	std::string toString(ActivationType act);
	ActivationType activationFromString(const std::string &str);

	class Layer
	{
		protected:
			std::weak_ptr<Context> m_context;
			std::weak_ptr<Tensor> m_workspace;

			std::vector<Shape> m_input_shapes;

			std::unique_ptr<Parameter> m_weights;
			std::unique_ptr<Parameter> m_bias;

			Initializer m_initializer;

			DataType m_dtype = DataType::FLOAT32;
			ActivationType m_activation;

			bool m_is_trainable = true;
			bool m_is_quantizable = true;
			int m_quantization_bits = 0;
			std::vector<AffineTransform> m_input_transforms;
			AffineTransform m_output_transform;
			Tensor m_channel_scales;
		public:
			Layer(std::string activation = "linear", DataType dtype = DataType::FLOAT32);

			Layer(const Layer &other) = delete;
			Layer(Layer &&other) = delete;
			Layer& operator=(const Layer &other) = delete;
			Layer& operator=(const Layer &&other) = delete;
			virtual ~Layer() = default;

			void setTrainable(bool b) noexcept;
			bool isTrainable() const noexcept;
			bool isQuantizable() const noexcept;

			ActivationType getActivationType() const noexcept;
			void setActivationType(ActivationType act) noexcept;

			virtual Layer& quantizable(bool b) noexcept;
			virtual void setupQuantization(const std::vector<AffineTransform> &input_transforms, const AffineTransform &output_transform, int bits);
			/**
			 * documentation
			 */
			virtual std::string name() const = 0;
			virtual Json getConfig() const;
			virtual void loadConfig(const Json &config);

			virtual Json saveParameters(SerializedObject &binary_data) const;
			virtual void loadParameters(const Json &json, const SerializedObject &binary_data);

			int numberOfInputs() const noexcept;
			void setInputShape(const Shape &shape);
			virtual void setInputShape(const std::vector<Shape> &shapes);
			const std::vector<Shape>& getInputShapes() const noexcept;
			Shape getInputShape(int index = 0) const;
			virtual Shape getOutputShape() const = 0;
			virtual Shape getWeightShape() const;
			virtual Shape getBiasShape() const;

			Device device() const;
			DataType dtype() const noexcept;
			const Context& context() const;

			Parameter& getWeights();
			Parameter& getBias();
			const Parameter& getWeights() const;
			const Parameter& getBias() const;

			virtual void convertTo(DataType newType);
			virtual std::unique_ptr<Layer> clone(const Json &config) const = 0;

			virtual int getWorkspaceSize() const noexcept;
			virtual void setWorkspace(std::shared_ptr<Tensor> &workspace);
			virtual void changeContext(std::shared_ptr<Context> &context);

			virtual void init();

			virtual void forward(const std::vector<Tensor> &input, Tensor &output) = 0;
			virtual void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta) = 0;
			std::shared_ptr<Context> get_context();
	};

	std::unique_ptr<Layer> loadLayer(const Json &json, const SerializedObject &binary_data);

} /* namespace ml */

#endif /* MINML_LAYERS_LAYER_HPP_ */
