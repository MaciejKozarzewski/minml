/*
 * InputBlock.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BLOCKS_INPUTBLOCK_HPP_
#define MINML_BLOCKS_INPUTBLOCK_HPP_

#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/Conv2D.hpp>

namespace ml
{

	class InputBlock
	{
			Conv2D conv;
			BatchNorm bn;
		public:
	};

} /* namespace ml */

#endif /* MINML_BLOCKS_INPUTBLOCK_HPP_ */
