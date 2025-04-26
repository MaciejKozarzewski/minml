/*
 * Optimizer.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_OPTIMIZER_HPP_
#define MINML_TRAINING_OPTIMIZER_HPP_

#include <minml/core/Tensor.hpp>

#include <vector>

namespace ml
{
	class Parameter;
}

namespace ml
{
	class Optimizer
	{
		public:
			Optimizer() noexcept = default;
			Optimizer(const Optimizer &other) = delete;
			Optimizer(Optimizer &&other) = delete;
			Optimizer& operator=(const Optimizer &other) = delete;
			Optimizer& operator=(const Optimizer &&other) = delete;
			virtual ~Optimizer() = default;

			virtual float getLearningRate() const noexcept = 0;
			virtual void setLearningRate(float lr) noexcept = 0;
			virtual int getSteps() const noexcept = 0;

			virtual void restart(const Context &context) noexcept = 0;
			virtual void moveTo(Device newDevice) = 0;
			virtual void convertTo(const Context &context, DataType newType) = 0;
			virtual void apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale) = 0;

			virtual Json serialize(SerializedObject &binary_data) const = 0;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data) = 0;
	};

	class RAdam: public Optimizer
	{
		private:
			std::vector<Tensor> m_fp32_weights;
			std::vector<Tensor> m_momentums;
			std::vector<Tensor> m_variances;

			float m_learning_rate = 0.001f;
			float m_beta1 = 0.9f;
			float m_beta2 = 0.999f;
			float m_weight_decay = 0.0f;
			int m_steps = 0;
		public:
			RAdam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0.0f);

			float getLearningRate() const noexcept;
			void setLearningRate(float lr) noexcept;
			int getSteps() const noexcept;

			void restart(const Context &context) noexcept;
			void moveTo(Device newDevice);
			void convertTo(const Context &context, DataType newType);
			void apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

	class Lion: public Optimizer
	{
		private:
			std::vector<Tensor> m_fp32_weights;
			std::vector<Tensor> m_momentums;

			float m_learning_rate = 0.001f;
			float m_beta1 = 0.9f;
			float m_beta2 = 0.999f;
			float m_weight_decay = 0.0f;
			int m_steps = 0;
		public:
			Lion(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.99f, float weightDecay = 0.0f);

			float getLearningRate() const noexcept;
			void setLearningRate(float lr) noexcept;
			int getSteps() const noexcept;

			void restart(const Context &context) noexcept;
			void moveTo(Device newDevice);
			void convertTo(const Context &context, DataType newType);
			void apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_OPTIMIZER_HPP_ */
