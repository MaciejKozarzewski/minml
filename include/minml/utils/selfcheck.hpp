/*
 * selfcheck.hpp
 *
 *  Created on: Jun 12, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_SELFCHECK_HPP_
#define MINML_UTILS_SELFCHECK_HPP_

namespace ml
{
	class Device;
}

namespace ml
{

	int checkDeviceDetection(Device device);

	int checkWinogradTransforms(Device device);

	int checkPoolingAndScaling(Device device);

	int checkDepthwiseConv2D(Device device);

	int checkMatrixMultiplication(Device device);

	int checkActivationFuncion(Device device);

} /* namespace ml */

#endif /* MINML_UTILS_SELFCHECK_HPP_ */
