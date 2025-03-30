/*
 * BatchNormalization.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_BATCHNORMALIZATION_HPP_
#define MINML_LAYERS_BATCHNORMALIZATION_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class BatchNormalization: public Layer
	{
			Tensor m_historical_stats;
			Tensor m_avg_var;
			int m_history_id = 0;
			int m_total_steps = 0;
			int m_history_size = 100;
			bool m_use_gamma = true;
			bool m_use_beta = true;

		public:
			BatchNormalization(std::string activation = "linear", bool useGamma = true, bool useBeta = true, int historySize = 100);

			BatchNormalization& useGamma(bool b) noexcept;
			BatchNormalization& useBeta(bool b) noexcept;
			BatchNormalization& historySize(int s) noexcept;

			bool isUsingGamma() const noexcept;
			bool isUsingBeta() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			Json saveParameters(SerializedObject &binary_data) const;
			void loadParameters(const Json &json, const SerializedObject &binary_data);

			std::unique_ptr<Layer> clone(const Json &config) const;

			void changeContext(std::shared_ptr<Context> &context);

			void init();
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
			void updateStatistics();
			Tensor& getStatistics();
	};

} /* namespace ml */

#endif /* MINML_LAYERS_BATCHNORMALIZATION_HPP_ */
