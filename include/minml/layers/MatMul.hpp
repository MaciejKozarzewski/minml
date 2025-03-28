/*
 * MatMul.hpp
 *
 *  Created on: Feb 13, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_MATMUL_HPP_
#define MINML_LAYERS_MATMUL_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class MatMul: public Layer
	{
			int m_rows = 0;
			int m_columns = 0;
			char m_input_mode = 'n';
			bool m_use_bias = true;
		public:
			MatMul(int rows, int columns, char inputMode, bool useBias = true);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_MATMUL_HPP_ */
