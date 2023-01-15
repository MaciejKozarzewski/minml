/*
 * ValueHeadBlock.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BLOCKS_VALUEHEADBLOCK_HPP_
#define MINML_BLOCKS_VALUEHEADBLOCK_HPP_

#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/Dense.hpp>

namespace ml
{

	class ValueHeadBlock
	{
			Conv2D conv1;
			BatchNorm bn1;
			Dense dense1;
			BatchNorm bn2;
			Dense dense2;
		public:
	};

} /* namespace ml */

#endif /* MINML_BLOCKS_VALUEHEADBLOCK_HPP_ */
