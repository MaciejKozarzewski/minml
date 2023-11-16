/*
 * kernel_table.hpp
 *
 *  Created on: Nov 15, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_OPENCL_KERNEL_TABLE_HPP_
#define BACKEND_OPENCL_KERNEL_TABLE_HPP_

#include <string>

namespace ml
{
	namespace opencl
	{
		std::string get_activation_kernels_source();
		std::string get_batchnorm_kernels_source();
		std::string get_conversion_kernels_source();
		std::string get_global_pooling_kernels_source();
		std::string get_training_kernels_source();
		std::string get_winograd_nonfused_kernels_source();

	} /* namespace opencl */
} /* namespace ml */

#endif /* BACKEND_OPENCL_KERNEL_TABLE_HPP_ */
