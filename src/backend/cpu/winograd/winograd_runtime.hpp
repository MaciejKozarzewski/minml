/*
 * winograd_runtime.hpp
 *
 *  Created on: Sep 30, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_WINOGRAD_WINOGRAD_RUNTIME_HPP_
#define BACKEND_CPU_WINOGRAD_WINOGRAD_RUNTIME_HPP_

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include <functional>

namespace ml
{
	class WinogradTransformRuntime
	{
		public:
			constexpr static size_t WorkspaceAlignment = 64; // [bytes]
			constexpr static size_t SimdSize = 512 / 8; // [bytes]

			using transform_function = std::function<void(const void *src[], void *dst[], void *workspace, int filters)>;
			using output_transform_function = std::function<void(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
					const void *bias, bool use_relu)>;

			int tile_size = 0;
			int kernel_size = 0;
			mlDataType_t dtype = DTYPE_UNKNOWN;
			transform_function input_transform;
			output_transform_function output_transform;
			transform_function weight_transform;
			transform_function gradient_transform;
			transform_function update_transform;

			WinogradTransformRuntime() noexcept = default;
			WinogradTransformRuntime(int tileSize, int kernelSize, mlDataType_t dtype) noexcept;

			void transformWeights(mlContext_t context, mlShape_t shape, const void *weights, void *matrices, bool invert) const;
			void transformInput(mlContext_t context, mlShape_t shape, const void *input, void *matrices) const;
			void transformOutput(mlContext_t context, mlShape_t shape, const void *matrices, void *output, const void *bias, const void *ext,
					bool use_relu) const;
			void transformGradient(mlContext_t context, mlShape_t shape, const void *gradient, void *matrices) const;
			void transformUpdate(mlContext_t context, mlShape_t shape, const void *matrices, void *update) const;
		private:
			void setup_default_transforms() noexcept;
	};

} /* namespace ml */

#endif /* BACKEND_CPU_WINOGRAD_WINOGRAD_RUNTIME_HPP_ */
