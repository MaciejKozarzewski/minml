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
		namespace kernels
		{
		std::string activations_backward =
#include "kernels/activations_backward.opencl"
		;
		std::string activations_forward =
#include "kernels/activations_forward.opencl"
		;
		std::string add_bias_act =
#include "kernels/add_bias_act.opencl"
		;
		std::string batchnorm =
#include "kernels/batchnorm.opencl"
		;
		std::string common =
#include "kernels/common.opencl"
		;
		std::string conversion =
#include "kernels/conversion.opencl"
		;
		std::string global_pooling =
#include "kernels/global_pooling.opencl"
		;
		std::string reductions =
#include "kernels/reductions.opencl"
		;
		std::string training =
#include "kernels/training.opencl"
		;
		std::string winograd_nonfused =
#include "kernels/winograd_nonfused.opencl"
		;

	}
/* namespace kernels */
} /* namespace opencl */
} /* namespace ml */

