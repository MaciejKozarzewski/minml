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
		private:
			Tensor m_momentum;
			Tensor m_variance;

			float m_learning_rate = 0.001f;
			float m_beta1 = 0.9f;
			float m_beta2 = 0.999f;
			int m_steps = 0;
			float m_weight_decay = 0.0f;
		public:
			Optimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weight_decay = 0.0f);

			float getLearningRate() const noexcept;
			void setLearningRate(float lr) noexcept;
			int getSteps() const noexcept;

			void restart(const Context &context) noexcept;
			void moveTo(Device newDevice);
			void apply(const Context &context, Parameter &param, float scale);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

	class RAdam
	{
		private:
			std::vector<Tensor> m_momentums;
			std::vector<Tensor> m_variances;

			float m_learning_rate = 0.001f;
			float m_beta1 = 0.9f;
			float m_beta2 = 0.999f;
			int m_steps = 0;
		public:
			RAdam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f);

			float getLearningRate() const noexcept;
			void setLearningRate(float lr) noexcept;
			int getSteps() const noexcept;

			void restart(const Context &context) noexcept;
			void moveTo(Device newDevice);
			void apply(const Context &context, std::vector<Tensor> &weights, const std::vector<Tensor> &gradients, float scale);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_OPTIMIZER_HPP_ */
