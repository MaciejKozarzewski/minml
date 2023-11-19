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
		namespace kernels
		{
			extern std::string activations_backward;
			extern std::string activations_forward;
			extern std::string add_bias_act;
			extern std::string batchnorm;
			extern std::string common;
			extern std::string conversion;
			extern std::string global_pooling;
			extern std::string indexers;
			extern std::string lines_and_tiles;
			extern std::string reductions;
			extern std::string training;
			extern std::string winograd_nonfused;

		} /* namespace kernels */
	} /* namespace opencl */
} /* namespace ml */

#endif /* BACKEND_OPENCL_KERNEL_TABLE_HPP_ */
