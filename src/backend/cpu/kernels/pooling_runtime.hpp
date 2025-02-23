/*
 * pooling_runtime.hpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_POOLING_RUNTIME_HPP_
#define BACKEND_CPU_KERNELS_POOLING_RUNTIME_HPP_

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include <algorithm>
#include <functional>
#include <iostream>

namespace ml
{

	class PoolingRuntime
	{
			const void *input_ptr = nullptr;
			void *output_ptr = nullptr;
			int batch_size = 0;
			int hw = 0;
			int channels = 0;
			mlDataType_t input_dtype = DTYPE_UNKNOWN;
			mlDataType_t output_dtype = DTYPE_UNKNOWN;
		public:
			void setInput(const void *ptr, mlDataType_t dtype, mlShape_t shape) noexcept
			{
				input_ptr = ptr;
				input_dtype = dtype;
				assert(shape.rank == 4);
				batch_size = shape.dim[0];
				hw = shape.dim[1] * shape.dim[2];
				channels = shape.dim[3];
			}
			void setOutput(void *ptr, mlDataType_t dtype) noexcept
			{
				output_ptr = ptr;
				output_dtype = dtype;
			}
			void run();
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_POOLING_RUNTIME_HPP_ */
