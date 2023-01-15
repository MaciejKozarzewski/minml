/*
 * ResidualBlock.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BLOCKS_RESIDUALBLOCK_HPP_
#define MINML_BLOCKS_RESIDUALBLOCK_HPP_

#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/Conv2D.hpp>

namespace ml
{

	class ResidualBlock
	{
			Conv2D conv1;
			BatchNorm bn1;
			Conv2D conv2;
			BatchNorm bn2;
		public:
	};

} /* namespace ml */



#endif /* MINML_BLOCKS_RESIDUALBLOCK_HPP_ */
