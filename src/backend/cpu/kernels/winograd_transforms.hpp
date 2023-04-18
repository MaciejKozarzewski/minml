/*
 * winograd_transforms.hpp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_WINOGRAD_TRANSFORMS_HPP_
#define BACKEND_CPU_KERNELS_WINOGRAD_TRANSFORMS_HPP_

#include "../vectors/vectors.hpp"

namespace SIMD_NAMESPACE
{

	template<int Length, typename T>
	struct Line
	{
		private:
			Vector<T> data[Length];
		public:
			template<typename U>
			inline void load_row(U **ptr, const int row, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * columns + i] + offset, num);
			}
			template<typename U>
			inline void load_row(const U **ptr, const int row, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[row * columns + i] + offset, num);
			}
			template<typename U>
			inline void store_row(U **ptr, const int row, const int offset, const int num, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].store(ptr[row * columns + i] + offset, num);
			}
			template<typename U>
			inline void load_column(const U **ptr, const int col, const int offset, const int num, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].load(ptr[i * columns + col] + offset, num);
			}
			template<typename U>
			inline void store_column(U **ptr, const int col, const int offset, const int num, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i].store(ptr[i * columns + col] + offset, num);
			}

			inline void load_row(const Vector<T> *ptr, const int row, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[row * columns + i];
			}
			inline void store_row(Vector<T> *ptr, const int row, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					ptr[row * columns + i] = data[i];
			}
			inline void load_column(const Vector<T> *ptr, const int col, int columns) noexcept
			{
				for (int i = 0; i < Length; i++)
					data[i] = ptr[i * columns + col];
			}
			inline void store_column(Vector<T> *ptr, const int col, int columns) const noexcept
			{
				for (int i = 0; i < Length; i++)
					ptr[i * columns + col] = data[i];
			}

			inline int length() const noexcept
			{
				return Length;
			}
			inline Vector<T>& operator[](int index) noexcept
			{
				assert(index >= 0 && index < Length);
				return data[index];
			}
			inline Vector<T> operator[](int index) const noexcept
			{
				assert(index >= 0 && index < Length);
				return data[index];
			}
	};

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
	};

	/*
	 * Kernel 3x3, tile 2x2
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 2, T>
	{
			inline Line<4, T> operator()(const Line<3, T> &line) const noexcept
			{
				Line<4, T> result;
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
			inline Line<4, T> operator()(const Line<4, T> &line) const noexcept
			{
				Line<4, T> result;
				result[0] = line[0] - line[2];
				result[1] = line[1] + line[2];
				result[2] = line[2] - line[1];
				result[3] = line[3] - line[1];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 2, T>
	{
			inline Line<2, T> operator()(const Line<4, T> &line) const noexcept
			{
				const Vector<T> c05(0.5);

				Line<2, T> result;
				result[0] = mul_add(c05, line[1] + line[2], line[0]);
				result[1] = mul_add(c05, line[1] - line[2], line[3]);
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 3, 2, T>
	{
			inline Line<4, T> operator()(const Line<2, T> &line) const noexcept
			{
				Line<4, T> result;
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
			inline Line<3, T> operator()(const Line<4, T> &line) const noexcept
			{
				const Vector<T> c05(0.5);

				Line<3, T> result;
				result[0] = mul_add(c05, line[1] + line[2], line[0]);
				result[1] = c05 * (line[1] - line[2]);
				result[2] = mul_add(c05, line[1] + line[2], line[3]);
				return result;
			}
	};

	/*
	 * Kernel 3x3, tile 4x4
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 3, 4, T>
	{
			inline Line<6, T> operator()(const Line<3, T> &line) const noexcept
			{
				const Vector<T> c13(1.0 / 3.0);
				const Vector<T> c23(2.0 / 3.0);
				const Vector<T> c2(2.0);

				Line<6, T> result;
				result[0] = line[0];
				result[1] = mul_add(c23, line[0] + line[2], c23 * line[1]);
				result[2] = mul_sub(c23, line[0] + line[2], c23 * line[1]);
				result[3] = mul_add(c13, line[0] + line[2], c23 * line[1]) + line[2];
				result[4] = mul_sub(c13, line[0] + line[2], c23 * line[1]) + line[2];
				result[5] = c2 * line[2];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 3, 4, T>
	{
			inline Line<6, T> operator()(const Line<6, T> &line) const noexcept
			{
				const Vector<T> c025(0.25);
				const Vector<T> c05(0.5);

				Line<6, T> result;
				result[0] = mul_add(c025, line[4] - line[2], line[0] - line[2]);
				result[1] = neg_mul_add(c025, line[3] + line[4], line[1] + line[2]);
				result[2] = mul_add(c025, line[3] - line[4], line[2] - line[1]);
				result[3] = mul_sub(c05, line[4] - line[2], line[1] - line[3]);
				result[4] = mul_add(c05, line[4] - line[2], line[1] - line[3]);
				result[5] = mul_add(c025, line[5] - line[3], line[1] - line[3]);
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 3, 4, T>
	{
			inline Line<4, T> operator()(const Line<6, T> &line) const noexcept
			{
				const Vector<T> c025(0.25);
				const Vector<T> c05(0.5);
				const Vector<T> c2(2.0);

				Line<4, T> result;
				result[0] = mul_add(c025, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c05, line[3] - line[4], line[1] - line[2]);
				result[2] = line[1] + line[2] + line[3] + line[4];
				result[3] = mul_add(c2, line[3] - line[4] + line[5], line[1] - line[2]);
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 3, 4, T>
	{
			inline Line<6, T> operator()(const Line<4, T> &line) const noexcept
			{
				const Vector<T> c13(1.0 / 3.0);
				const Vector<T> c23(2.0 / 3.0);
				const Vector<T> c2(2.0);

				Line<6, T> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3]);
				result[3] = mul_add(c13, line[0] + line[2], line[2]) + mul_add(c23, line[1] + line[3], c2 * line[3]);
				result[4] = mul_add(c13, line[0] + line[2], line[2]) - mul_add(c23, line[1] + line[3], c2 * line[3]);
				result[5] = c2 * line[3];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::UPDATE, 3, 4, T>
	{
			inline Line<3, T> operator()(const Line<6, T> &line) const noexcept
			{
				const Vector<T> c025(0.25);
				const Vector<T> c05(0.5);
				const Vector<T> c2(2.0);

				Line<3, T> result;
				result[0] = mul_add(c025, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c05, line[3] - line[4], line[1] - line[2]);
				result[2] = mul_add(c2, line[5], line[1] + line[2] + line[3] + line[4]);
				return result;
			}
	};

	/*
	 * Kernel 5x5, tile 2x2
	 */
	template<typename T>
	struct Transform<TransformType::WEIGHT, 5, 2, T>
	{
			inline Line<6, T> operator()(const Line<5, T> &line) const noexcept
			{
				const Vector<T> c16(1.0 / 6.0);
				const Vector<T> c13(1.0 / 3.0);
				const Vector<T> c23(2.0 / 3.0);
				const Vector<T> c2(2.0);

				Line<6, T> result;
				result[0] = line[0];
				result[1] = c23 * (line[0] + line[1] + line[2] + line[3] + line[4]);
				result[2] = c23 * (line[0] - line[1] + line[2] - line[3] + line[4]);
				result[3] = c16 * line[0] + mul_add(c13, line[1] + line[3], line[3]) + mul_add(c23, line[2] + line[4], c2 * line[4]);
				result[4] = c16 * line[0] - mul_add(c13, line[1] + line[3], line[3]) + mul_add(c23, line[2] + line[4], c2 * line[4]);
				result[5] = c2 * line[4];
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::INPUT, 5, 2, T>
	{
			inline Line<6, T> operator()(const Line<6, T> &line) const noexcept
			{
				return Transform<TransformType::INPUT, 3, 4, T>()(line); // it turns out that those two transforms are the same
			}
	};
	template<typename T>
	struct Transform<TransformType::OUTPUT, 5, 2, T>
	{
			inline Line<2, T> operator()(const Line<6, T> &line) const noexcept
			{
				const Vector<T> c05(0.5);
				const Vector<T> c2(2.0);

				Line<2, T> result;
				result[0] = mul_add(c05, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c2, line[5], line[1] - line[2] + line[3] - line[4]);
				return result;
			}
	};
	template<typename T>
	struct Transform<TransformType::GRADIENT, 5, 2, T>
	{
			inline Line<6, T> operator()(const Line<2, T> &line) const noexcept
			{
				const Vector<T> c13(1.0 / 3.0);
				const Vector<T> c23(2.0 / 3.0);

				Line<6, T> result;
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
			inline Line<5, T> operator()(const Line<6, T> &line) const noexcept
			{
				const Vector<T> c025(0.25);
				const Vector<T> c05(0.5);
				const Vector<T> c2(2.0);
				const Vector<T> c4(4.0);

				Line<5, T> result;
				result[0] = mul_add(c025, line[3] + line[4], line[0] + line[1] + line[2]);
				result[1] = mul_add(c05, line[3] - line[4], line[1] - line[2]);
				result[2] = line[1] + line[2] + line[3] + line[4];
				result[3] = mul_add(c2, line[3] - line[4], line[1] - line[2]);
				result[4] = mul_add(c4, line[3] + line[4] + line[5], line[1] + line[2]);
				return result;
			}
	};
}
#endif /* BACKEND_CPU_KERNELS_WINOGRAD_TRANSFORMS_HPP_ */
