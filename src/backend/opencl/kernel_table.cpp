/*
 * kernel_table.cpp
 *
 *  Created on: Nov 15, 2023
 *      Author: Maciej Kozarzewski
 */

#include "kernel_table.hpp"

namespace ml
{
	namespace opencl
	{

		std::string get_activation_kernels_source()
		{

		}
		std::string get_batchnorm_kernels_source()
		{
		}
		std::string get_conversion_kernels_source()
		{
			const char* c =
#include "kernels/conversion.opencl"
					;
			return std::string(c);
		}
		std::string get_global_pooling_kernels_source()
		{
		}
		std::string get_training_kernels_source()
		{
		}
		std::string get_winograd_nonfused_kernels_source()
		{
		}

	} /* namespace opencl */
} /* namespace ml */

