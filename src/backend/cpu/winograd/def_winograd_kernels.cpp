/*
 * def_winograd_kernels.cpp
 *
 *  Created on: Jun 26, 2023
 *      Author: Maciej Kozarzewski
 */

#include "winograd_kernels.hpp"
#include "../vectors/types.hpp"
#include "../fp16.hpp"

#include <array>
#include <cinttypes>

namespace
{
	using namespace ml;

	template<typename SrcT, typename DstT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	float16 convert(float x) noexcept
	{
		return float16 { cpu::convert_fp32_to_fp16(x) };
	}
	template<>
	float convert(float16 x) noexcept
	{
		return cpu::convert_fp16_to_fp32(x.m_data);
	}
	float relu(float x) noexcept
	{
		return (x > 0.0f) ? x : 0.0f;
	}

	enum class TransformType
	{
		WEIGHT,
		INPUT,
		OUTPUT,
		GRADIENT,
		UPDATE
	};

	template<TransformType Type, int KernelSize, int TransformSize>
	constexpr int input_size() noexcept
	{
		switch (Type)
		{
			case TransformType::WEIGHT:
				return KernelSize;
			case TransformType::INPUT:
				return KernelSize + TransformSize - 1;
			case TransformType::OUTPUT:
				return KernelSize + TransformSize - 1;
			case TransformType::GRADIENT:
				return TransformSize;
			case TransformType::UPDATE:
				return KernelSize + TransformSize - 1;
			default:
				return 0;
		}
	}
	template<TransformType Type, int KernelSize, int TransformSize>
	constexpr int output_size() noexcept
	{
		switch (Type)
		{
			case TransformType::WEIGHT:
				return KernelSize + TransformSize - 1;
			case TransformType::INPUT:
				return KernelSize + TransformSize - 1;
			case TransformType::OUTPUT:
				return TransformSize;
			case TransformType::GRADIENT:
				return KernelSize + TransformSize - 1;
			case TransformType::UPDATE:
				return KernelSize;
			default:
				return 0;
		}
	}

	template<TransformType Type, int KernelSize, int TransformSize, typename T>
	struct Transform
	{
	};

	/*
	 * Kernel 3x3, tile 2x2
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 2, T>
	{
			inline std::array<T, 4> operator()(const std::array<T, 3> &line) const noexcept
			{
				std::array<T, 4> result;
				result[0] = line[0];
				result[1] = line[0] + line[1] + line[2];
				result[2] = line[0] - line[1] + line[2];
				result[3] = line[2];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 3, 2, T>
	{
			inline std::array<T, 4> operator()(const std::array<T, 4> &line) const noexcept
			{
				std::array<T, 4> result;
				result[0] = line[0] - line[2];
				result[1] = line[1] + line[2];
				result[2] = -line[1] + line[2];
				result[3] = -line[1] + line[3];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 2, T>
	{
			inline std::array<T, 2> operator()(const std::array<T, 4> &line) const noexcept
			{
				const T c05 = 0.5;

				std::array<T, 2> result;
				result[0] = line[0] + c05 * (line[1] + line[2]);
				result[1] = c05 * (line[1] - line[2]) + line[3];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 3, 2, T>
	{
			inline std::array<T, 4> operator()(const std::array<T, 2> &line) const noexcept
			{
				std::array<T, 4> result;
				result[0] = line[0];
				result[1] = line[0] + line[1];
				result[2] = line[0] - line[1];
				result[3] = line[1];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 3, 2, T>
	{
			inline std::array<T, 3> operator()(const std::array<T, 4> &line) const noexcept
			{
				const T c05 = 0.5;

				std::array<T, 3> result;
				result[0] = line[0] + c05 * (line[1] + line[2]);
				result[1] = c05 * (line[1] - line[2]);
				result[2] = c05 * (line[1] + line[2]) + line[3];
				return result;
			}
	};

	/*
	 * Kernel 3x3, tile 4x4
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 4, T>
	{
			inline std::array<T, 6> operator()(const std::array<T, 3> &line) const noexcept
			{
				const T c13 = 1.0 / 3.0;
				const T c23 = 2.0 / 3.0;
				const T c43 = 4.0 / 3.0;
				const T c2 = 2.0;

				std::array<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2]);
				result[2] = c23 * (line[0] - line[1] + line[2]);
				result[3] = c13 * line[0] + c23 * line[2] + c43 * line[2];
				result[4] = c13 * line[0] - c23 * line[2] + c43 * line[2];
				result[5] = c2 * line[2];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 3, 4, T>
	{
			inline std::array<T, 6> operator()(const std::array<T, 6> &line) const noexcept
			{
				std::array<T, 6> result;
				result[0] = line[0] - 1.25 * line[2] + 0.25 * line[4];
				result[1] = line[1] + line[2] - 0.25 * line[3] - 0.25 * line[4];
				result[2] = -line[1] + line[2] + 0.25 * line[3] - 0.25 * line[4];
				result[3] = -line[1] - 0.5 * line[2] + line[3] + 0.5 * line[4];
				result[4] = line[1] - 0.5 * line[2] - line[3] + 0.5 * line[4];
				result[5] = line[1] - 1.25 * line[3] + 0.25 * line[5];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 4, T>
	{
			inline std::array<T, 4> operator()(const std::array<T, 6> &line) const noexcept
			{
				std::array<T, 4> result;
				result[0] = line[0] + line[2] + line[3] + 0.25 * (line[3] + line[4]);
				result[1] = line[1] - line[2] + 0.5 * (line[3] - line[4]);
				result[2] = line[1] + line[2] + line[3] + line[4];
				result[3] = line[1] - line[2] + 2 * (line[3] - line[4] + line[5]);
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 3, 4, T>
	{
			inline std::array<T, 6> operator()(const std::array<T, 4> &line) const noexcept
			{
				const T c13 = 1.0 / 3.0;
				const T c23 = 2.0 / 3.0;
				const T c43 = 4.0 / 3.0;
				const T c83 = 8.0 / 3.0;
				const T c2 = 2.0;

				std::array<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3]);
				result[3] = c13 * line[0] + c23 * line[1] + c43 * line[2] + c83 * line[3];
				result[4] = c13 * line[0] - c23 * line[1] + c43 * line[2] - c83 * line[3];
				result[5] = c2 * line[3];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 3, 4, T>
	{
			inline std::array<T, 3> operator()(const std::array<T, 6> &line) const noexcept
			{
				std::array<T, 3> result;
				result[0] = line[0] + line[2] + line[3] + 0.25 * (line[3] + line[4]);
				result[1] = line[1] - line[2] + 0.5 * (line[3] - line[4]);
				result[2] = line[1] + line[2] + line[3] + line[4] + 2 * line[5];
				return result;
			}
	};

	/*
	 * Kernel 3x3, tile 5x5
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 5, T>
	{
			inline std::array<T, 7> operator()(const std::array<T, 3> &line) const noexcept
			{
				const T c2_3 = 2.0 / 3.0;
				const T c2_9 = 2.0 / 9.0;
				const T c4_9 = 4.0 / 9.0;
				const T c4_15 = 4.0 / 15.0;
				const T c16_45 = 16.0 / 45.0;
				const T c200 = 2.0;
				const T c050 = 0.5;

				std::array<T, 7> result;
				result[0] = line[0];
				result[1] = c2_3 * (line[0] + line[1] + line[2]);
				result[2] = c2_9 * (line[0] - line[1] + line[2]);
				result[3] = c4_9 * (c050 * line[0] + line[1] + c200 * line[2]);
				result[4] = c4_15 * (c050 * line[0] - line[1] + c200 * line[2]);
				result[5] = c16_45 * (c200 * line[0] + line[1] + c050 * line[2]);
				result[6] = c200 * line[2];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 3, 5, T>
	{
			inline std::array<T, 7> operator()(const std::array<T, 7> &line) const noexcept
			{
				std::array<T, 7> result;
				result[0] = line[0] - 2 * line[1] - 1.25 * line[2] + 2.5 * line[3] + 0.25 * line[4] - 0.5 * line[5];
				result[1] = -line[1] + line[2] + 2.25 * line[3] - 0.25 * line[4] - 0.5 * line[5];
				result[2] = -line[1] + 3 * line[2] - 1.75 * line[3] - 0.75 * line[4] + 0.5 * line[5];
				result[3] = 0.5 * line[1] - 0.75 * line[2] - line[3] + 0.75 * line[4] + 0.5 * line[5];
				result[4] = 0.5 * line[1] - 1.25 * line[2] + 1.25 * line[4] - 0.5 * line[5];
				result[5] = line[1] - 1.25 * line[3] + 0.25 * line[5];
				result[6] = -line[1] + 2 * line[2] + 1.25 * line[3] - 2.5 * line[4] - 0.25 * line[5] + 0.5 * line[6];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 5, T>
	{
			inline std::array<T, 5> operator()(const std::array<T, 7> &line) const noexcept
			{
				std::array<T, 5> result;
				result[0] = line[0] + line[1] + line[2] + 0.25 * (line[3] + line[4]) + 4 * line[5];
				result[1] = line[1] - line[2] + 0.5 * (line[3] - line[4]) + 2 * line[5];
				result[2] = line[1] + line[2] + line[3] + line[4] + line[5];
				result[3] = line[1] - line[2] + 2 * (line[3] - line[4]) + 0.5 * line[5];
				result[4] = line[1] + line[2] + 4 * (line[3] + line[4]) + 0.25 * line[5] + line[6];
				return result;
			}
	};

	/*
	 * Kernel 5x5, tile 2x2
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 5, 2, T>
	{
			inline std::array<T, 6> operator()(const std::array<T, 5> &line) const noexcept
			{
				const T c16 = 1.0 / 6.0;
				const T c13 = 1.0 / 3.0;
				const T c23 = 2.0 / 3.0;
				const T c2 = 2.0;

				std::array<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3] + line[4]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3] + line[4]);
				result[3] = c16 * line[0] + c13 * (line[1] + line[3]) + line[3] + c23 * (line[2] + line[4]) + c2 * line[4];
				result[4] = c16 * line[0] - c13 * (line[1] + line[3]) + line[3] + c23 * (line[2] + line[4]) + c2 * line[4];
				result[5] = c2 * line[4];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 5, 2, T>
	{
			inline std::array<T, 6> operator()(const std::array<T, 6> &line) const noexcept
			{
				return Transform<TransformType::INPUT, 3, 4, T>()(line); // it turns out that those two transforms are the same
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 5, 2, T>
	{
			inline std::array<T, 2> operator()(const std::array<T, 6> &line) const noexcept
			{
				std::array<T, 2> result;
				result[0] = line[0] + line[1] + line[2] + 0.5 * (line[3] + line[4]);
				result[1] = line[1] - line[2] + line[3] - line[4] + 2.0 * line[5];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 5, 2, T>
	{
			inline std::array<T, 6> operator()(const std::array<T, 2> &line) const noexcept
			{
				const T c13 = 1.0 / 3.0;
				const T c23 = 2.0 / 3.0;

				std::array<T, 6> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1]);
				result[2] = c23 * (line[0] - line[1]);
				result[3] = c13 * line[0] + c23 * line[1];
				result[4] = c13 * line[0] - c23 * line[1];
				result[5] = line[1];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 5, 2, T>
	{
			inline std::array<T, 5> operator()(const std::array<T, 6> &line) const noexcept
			{
				std::array<T, 5> result;
				result[0] = line[0] + line[1] + line[2] + 0.25 * (line[3] + line[4]);
				result[1] = line[1] - line[2] + 0.5 * (line[3] - line[4]);
				result[2] = line[1] + line[2] + line[3] + line[4];
				result[3] = line[1] - line[2] + 2.0 * (line[3] - line[4]);
				result[4] = line[1] + line[2] + 4.0 * (line[3] + line[4] + line[5]);
				return result;
			}
	};

	template<TransformType Type, int KernelSize, int TransformSize, typename DataType, typename ComputeType>
	void impl_transform(const void *src[], void *dst[], void *workspace, int filters, const void *ext[] = nullptr, const void *bias = nullptr,
			bool use_relu = false)
	{
		constexpr int in_size = input_size<Type, KernelSize, TransformSize>();
		constexpr int out_size = output_size<Type, KernelSize, TransformSize>();

		const Transform<Type, KernelSize, TransformSize, ComputeType> transform;

		for (int f = 0; f < filters; f++)
		{
			std::array<ComputeType, in_size> input_tile;

			for (int col = 0; col < in_size; col++)
			{
				for (int row = 0; row < in_size; row++)
					input_tile[row] = convert<DataType, ComputeType>(reinterpret_cast<const DataType*>(src[row * in_size + col])[f]);
				const std::array<ComputeType, out_size> transformed = transform(input_tile);
				for (int row = 0; row < out_size; row++)
					reinterpret_cast<ComputeType*>(workspace)[row * in_size + col] = transformed[row];
			}

			for (int row = 0; row < out_size; row++)
			{
				for (int col = 0; col < in_size; col++)
					input_tile[col] = reinterpret_cast<ComputeType*>(workspace)[row * in_size + col];

				const std::array<ComputeType, out_size> transformed = transform(input_tile);
				for (int col = 0; col < out_size; col++)
				{
					ComputeType tmp = transformed[col];
					if (Type == TransformType::OUTPUT)
					{
						if (ext != nullptr)
							tmp += convert<DataType, ComputeType>(reinterpret_cast<const DataType*>(ext[row * out_size + col])[f]);
						if (bias != nullptr)
							tmp += convert<DataType, ComputeType>(reinterpret_cast<const DataType*>(bias)[f]);
						if (use_relu)
							tmp = relu(tmp);
					}
					reinterpret_cast<DataType*>(dst[row * out_size + col])[f] = convert<ComputeType, DataType>(tmp);
				}
			}
		}
	}
}

namespace ml
{
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_weight_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::WEIGHT, 3, 4, float, float>(src, dst, workspace, filters);
	}
	void winograd_input_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::INPUT, 3, 4, float, float>(src, dst, workspace, filters);
	}
	void winograd_output_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
		impl_transform<TransformType::OUTPUT, 3, 4, float, float>(src, dst, workspace, filters, ext, bias, use_relu);
	}
	void winograd_gradient_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::GRADIENT, 3, 4, float, float>(src, dst, workspace, filters);
	}
	void winograd_update_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::UPDATE, 3, 4, float, float>(src, dst, workspace, filters);
	}
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_weight_transform_4x4_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::WEIGHT, 3, 4, float16, float>(src, dst, workspace, filters);
	}
	void winograd_input_transform_4x4_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::INPUT, 3, 4, float16, float>(src, dst, workspace, filters);
	}
	void winograd_output_transform_4x4_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
		impl_transform<TransformType::OUTPUT, 3, 4, float16, float>(src, dst, workspace, filters, ext, bias, use_relu);
	}

	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_weight_transform_5x5_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::WEIGHT, 3, 5, float, float>(src, dst, workspace, filters);
	}
	void winograd_input_transform_5x5_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::INPUT, 3, 5, float, float>(src, dst, workspace, filters);
	}
	void winograd_output_transform_5x5_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
		impl_transform<TransformType::OUTPUT, 3, 5, float, float>(src, dst, workspace, filters, ext, bias, use_relu);
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_weight_transform_5x5_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::WEIGHT, 3, 5, float16, float>(src, dst, workspace, filters);
	}
	void winograd_input_transform_5x5_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::INPUT, 3, 5, float16, float>(src, dst, workspace, filters);
	}
	void winograd_output_transform_5x5_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
		impl_transform<TransformType::OUTPUT, 3, 5, float16, float>(src, dst, workspace, filters, ext, bias, use_relu);
	}

	/*
	 * Transforms for 5x5 kernel and 2x2 tile size in FP32
	 */
	void winograd_weight_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::WEIGHT, 5, 2, float, float>(src, dst, workspace, filters);
	}
	void winograd_input_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::INPUT, 5, 2, float, float>(src, dst, workspace, filters);
	}
	void winograd_output_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
		impl_transform<TransformType::OUTPUT, 5, 2, float, float>(src, dst, workspace, filters, ext, bias, use_relu);
	}
	void winograd_gradient_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::GRADIENT, 5, 2, float, float>(src, dst, workspace, filters);
	}
	void winograd_update_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		impl_transform<TransformType::UPDATE, 5, 2, float, float>(src, dst, workspace, filters);
	}

} /* namespace ml */

