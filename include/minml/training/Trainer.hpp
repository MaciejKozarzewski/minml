/*
 * Trainer.hpp
 *
 *  Created on: Mar 23, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_TRAINING_TRAINER_HPP_
#define MINML_TRAINING_TRAINER_HPP_

#include <minml/core/Tensor.hpp>
#include <minml/training/LossFunction.hpp>
#include <minml/training/GradientScaler.hpp>
#include <minml/training/Optimizer.hpp>
#include <minml/training/Regularizer.hpp>

#include <vector>

namespace ml
{
	class Graph;
}

namespace ml
{

	class Trainer
	{
			Graph *m_graph_ptr = nullptr;
			std::vector<Tensor> m_targets;
			std::vector<Tensor> m_masks;
			std::vector<std::unique_ptr<LossFunction>> m_losses;
			std::vector<float> m_loss_weights;

			RAdam m_optimizer;
			RegularizerL2 m_regularizer;
			std::vector<Tensor> m_fp32_weights_copy;

			DataType m_training_dtype = DataType::FLOAT32;
		public:
			Trainer(Graph &graph, DataType dtype);

			Graph& graph();
			void setLossFunction(int index, const LossFunction &loss, float weight = 1.0f);
			void setOptimizer(const RAdam &opt);
			void setRegularizer(const RegularizerL2 &reg);

			const Tensor& getTarget(int index) const;
			const Tensor& getMask(int index) const;
			Tensor& getTarget(int index);
			Tensor& getMask(int index);

			void moveTo(Device newDevice);
			void train(int batchSize, GradientScaler &gradientScaler);
			std::vector<float> getLoss(int batchSize);
	};

} /* namespace ml */

#endif /* MINML_TRAINING_TRAINER_HPP_ */
