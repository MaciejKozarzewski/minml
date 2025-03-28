/*
 * GlobalBroadcastHW.hpp
 *
 *  Created on: Feb 16, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_GLOBALBROADCASTHW_HPP_
#define MINML_LAYERS_GLOBALBROADCASTHW_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class GlobalBroadcastHW: public Layer
	{
			bool m_use_bias = true;
		public:
			GlobalBroadcastHW(std::string activation = "linear", bool use_bias = true);

			GlobalBroadcastHW& useBias(bool b) noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			int getWorkspaceSize() const noexcept;
			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_GLOBALBROADCASTHW_HPP_ */
