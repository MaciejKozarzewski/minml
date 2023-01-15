/*
 * winograd_transforms.cuh
 *
 *  Created on: Jan 6, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_KERNELS_WINOGRAD_TRANSFORMS_CUH_
#define BACKEND_CUDA_KERNELS_WINOGRAD_TRANSFORMS_CUH_

#include "../helpers/lines_and_tiles.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

namespace ml
{
	enum class TransformType
	{
		WEIGHT,
		INPUT,
		OUTPUT,
		GRADIENT,
		UPDATE
	};

	template<TransformType Type, int KernelSize, int TransformSize, typename T>
	struct Transform
	{
			__device__ T operator()(const int row, const Line<T, KernelSize + TransformSize - 1> &column) const
			{
				return static_cast<T>(0);
			}
	};

	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 3> &column) const
			{
				assert(0 <= row && row < 4);
				switch (row)
				{
					case 0:
						return column.x0;
					case 1:
						return static_cast<T>(0.5f) * (column.x0 + column.x1 + column.x2);
					case 2:
						return static_cast<T>(0.5f) * (column.x0 - column.x1 + column.x2);
					case 3:
						return column.x2;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 3, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 4> &column) const
			{
				assert(0 <= row && row < 4);
				switch (row)
				{
					case 0:
						return column.x0 - column.x2;
					case 1:
						return column.x1 + column.x2;
					case 2:
						return column.x2 - column.x1;
					case 3:
						return column.x3 - column.x1;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 4> &column) const
			{
				assert(0 <= row && row < 2);
				switch (row)
				{
					case 0:
						return column.x0 + column.x1 + column.x2;
					case 1:
						return column.x1 - column.x2 + column.x3;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 3, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 2> &column) const
			{
				assert(0 <= row && row < 4);
				switch (row)
				{
					case 0:
						return column.x0;
					case 1:
						return column.x0 + column.x1;
					case 2:
						return column.x0 - column.x1;
					case 3:
						return column.x1;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 3, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 4> &column) const
			{
				assert(0 <= row && row < 3);
				const T c05 = 0.5f;
				switch (row)
				{
					case 0:
						return column.x0 + c05 * (column.x1 + column.x2);
					case 1:
						return c05 * (column.x1 - column.x2);
					case 2:
						return c05 * (column.x1 + column.x2) + column.x3;
				}
			}
	};

	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 4, T>
	{
			__device__ T operator()(const int row, const Line<T, 3> &column) const
			{
				assert(0 <= row && row < 6);
				const T c13 = 1.0f / 3.0f;
				const T c23 = 2.0f / 3.0f;
				switch (row)
				{
					case 0:
						return column.x0;
					case 1:
						return c23 * (column.x0 + column.x1 + column.x2);
					case 2:
						return c23 * (column.x0 - column.x1 + column.x2);
					case 3:
						return c13 * (column.x0 + static_cast<T>(2.0f) * column.x1 + static_cast<T>(4.0f) * column.x2);
					case 4:
						return c13 * (column.x0 - static_cast<T>(2.0f) * column.x1 + static_cast<T>(4.0f) * column.x2);
					case 5:
						return static_cast<T>(2.0f) * column.x2;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 3, 4, T>
	{
			__device__ T operator()(const int row, const Line<T, 6> &column) const
			{
				assert(0 <= row && row < 6);
				switch (row)
				{
					case 0:
						return column.x0 - column.x2 + static_cast<T>(0.25f) * (column.x4 - column.x2);
					case 1:
						return column.x1 + column.x2 - static_cast<T>(0.25f) * (column.x3 + column.x4);
					case 2:
						return column.x2 - column.x1 + static_cast<T>(0.25f) * (column.x3 - column.x4);
					case 3:
						return column.x3 - column.x1 + static_cast<T>(0.5f) * (column.x4 - column.x2);
					case 4:
						return column.x1 - column.x3 + static_cast<T>(0.5f) * (column.x4 - column.x2);
					case 5:
						return column.x1 - column.x3 + static_cast<T>(0.25f) * (column.x5 - column.x3);
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 4, T>
	{
			__device__ T operator()(const int row, const Line<T, 6> &column) const
			{
				assert(0 <= row && row < 4);
				switch (row)
				{
					case 0:
						return column.x0 + column.x1 + column.x2 + static_cast<T>(0.25f) * (column.x3 + column.x4);
					case 1:
						return column.x1 - column.x2 + static_cast<T>(0.5f) * (column.x3 - column.x4);
					case 2:
						return column.x1 + column.x2 + column.x3 + column.x4;
					case 3:
						return column.x1 - column.x2 + static_cast<T>(2.0f) * (column.x3 - column.x4 + column.x5);
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 3, 4, T>
	{
			__device__ T operator()(const int row, const Line<T, 4> &column) const
			{
				assert(0 <= row && row < 6);
				const T c13 = 1.0f / 3.0f;
				const T c23 = 2.0f / 3.0f;
				switch (row)
				{
					case 0:
						return column.x0;
					case 1:
						return c23 * (column.x0 + column.x1 + column.x2 + column.x3);
					case 2:
						return c23 * (column.x0 - column.x1 + column.x2 - column.x3);
					case 3:
						return c13 * (column.x0 + 2.0f * column.x1 + 4.0f * column.x2 + 8.0f * column.x3);
					case 4:
						return c13 * (column.x0 - 2.0f * column.x1 + 4.0f * column.x2 - 8.0f * column.x3);
					case 5:
						return static_cast<T>(2.0f) * column.x3;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 3, 4, T>
	{
			__device__ T operator()(const int row, const Line<T, 6> &column) const
			{
				assert(0 <= row && row < 3);
				switch (row)
				{
					case 0:
						return column.x0 + column.x1 + column.x2 + static_cast<T>(0.25f) * (column.x3 + column.x4);
					case 1:
						return column.x1 - column.x2 + static_cast<T>(0.5f) * (column.x3 - column.x4);
					case 2:
						return column.x1 + column.x2 + column.x3 + column.x4 + 2.0f * column.x5;
				}
			}
	};

	template<typename T>
	struct Transform<TransformType::WEIGHT, 5, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 5> &column) const
			{
				assert(0 <= row && row < 6);
				const T c16 = 1.0f / 6.0f;
				const T c23 = 2.0f / 3.0f;
				switch (row)
				{
					case 0:
						return column.x0;
					case 1:
						return c23 * (column.x0 + column.x1 + column.x2 + column.x3 + column.x4);
					case 2:
						return c23 * (column.x0 - column.x1 + column.x2 - column.x3 + column.x4);
					case 3:
						return c16
								* (column.x0 + static_cast<T>(2.0f) * column.x1 + static_cast<T>(4.0f) * column.x2 + static_cast<T>(8.0f) * column.x3
										+ static_cast<T>(16.0f) * column.x4);
					case 4:
						return c16
								* (column.x0 - static_cast<T>(2.0f) * column.x1 + static_cast<T>(4.0f) * column.x2 - static_cast<T>(8.0f) * column.x3
										+ static_cast<T>(16.0f) * column.x4);
					case 5:
						return static_cast<T>(2.0f) * column.x4;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 5, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 6> &column) const
			{
				return Transform<TransformType::INPUT, 3, 4, T>()(row, column); // 2x2 input transform for 5x5 kernel is the same as 4x4 transform for 3x3 kernel
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 5, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 6> &column) const
			{
				assert(0 <= row && row < 2);
				switch (row)
				{
					case 0:
						return column.x0 + column.x1 + column.x2 + static_cast<T>(0.5f) * (column.x3 + column.x4);
					case 1:
						return column.x1 - column.x2 + column.x3 - column.x4 + static_cast<T>(2.0f) * column.x5;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 5, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 2> &column) const
			{
				assert(0 <= row && row < 6);
				const T c13 = 1.0f / 3.0f;
				const T c23 = 2.0f / 3.0f;
				switch (row)
				{
					case 0:
						return column.x0;
					case 1:
						return c23 * (column.x0 + column.x1);
					case 2:
						return c23 * (column.x0 - column.x1);
					case 3:
						return c13 * (column.x0 + static_cast<T>(2.0f) * column.x1);
					case 4:
						return c13 * (column.x0 - static_cast<T>(2.0f) * column.x1);
					case 5:
						return column.x1;
				}
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 5, 2, T>
	{
			__device__ T operator()(const int row, const Line<T, 6> &column) const
			{
				assert(0 <= row && row < 5);
				switch (row)
				{
					case 0:
						return column.x0 + column.x1 + column.x2 + static_cast<T>(0.25f) * (column.x3 + column.x4);
					case 1:
						return column.x1 - column.x2 + static_cast<T>(0.5f) * (column.x3 - column.x4);
					case 2:
						return column.x1 + column.x2 + column.x3 + column.x4;
					case 3:
						return column.x1 - column.x2 + 2.0f * (column.x3 - column.x4);
					case 4:
						return column.x1 + column.x2 + 4.0f * (column.x3 + column.x4 + column.x5);
				}
			}
	};

} /* namespace ml */

#endif /* BACKEND_CUDA_KERNELS_WINOGRAD_TRANSFORMS_CUH_ */
