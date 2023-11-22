/*
 * testing_util.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/testing_util.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/DataType.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>

#include <complex>

namespace
{
	using namespace ml;
	void init_for_test_fp32(float *ptr, size_t length, float shift, float scale)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = sin(i / 10.0f + shift) * scale;
	}
	void init_for_test_fp16(uint16_t *ptr, size_t length, float shift, float scale)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = convert_fp32_to_fp16(sinf(i / 10.0f + shift) * scale);
	}

	double diff_for_test_fp32(const float *ptr1, const float *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += fabs(ptr1[i] - ptr2[i]);
		return result / length;
	}
	double diff_for_test_fp16(const uint16_t *ptr1, const uint16_t *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += fabs(convert_fp16_to_fp32(ptr1[i]) - convert_fp16_to_fp32(ptr2[i]));
		return result / length;
	}

	double norm_for_test_fp32(const float *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += fabs(ptr[i]);
		return result;
	}
	double norm_for_test_fp16(const uint16_t *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += fabs(convert_fp16_to_fp32(ptr[i]));
		return result;
	}

	double sum_for_test_fp32(const float *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += ptr[i];
		return result;
	}
	double sum_for_test_fp16(const uint16_t *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += convert_fp16_to_fp32(ptr[i]);
		return result;
	}

	void abs_for_test_fp32(float *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = fabs(ptr[i]);
	}
	void abs_for_test_fp16(uint16_t *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = fabsf(convert_fp16_to_fp32(ptr[i]));
	}
}

namespace ml
{
	namespace testing
	{
		void initForTest(Tensor &t, double shift, double scale)
		{
			Tensor tmp(t.shape(), t.dtype(), Device::cpu());
			switch (tmp.dtype())
			{
				case DataType::FLOAT16:
					init_for_test_fp16(reinterpret_cast<uint16_t*>(tmp.data()), tmp.volume(), shift, scale);
					break;
				case DataType::FLOAT32:
					init_for_test_fp32(reinterpret_cast<float*>(tmp.data()), tmp.volume(), shift, scale);
					break;
				default:
					throw DataTypeNotSupported(METHOD_NAME, tmp.dtype());
			}
			t.copyFrom(Context(), tmp);
		}
		double diffForTest(const Tensor &lhs, const Tensor &rhs)
		{
			assert(lhs.shape() == rhs.shape());
			assert(lhs.dtype() == rhs.dtype());

			if (lhs.volume() == 0)
				return 0.0;

			Tensor tmp_lhs(lhs.shape(), lhs.dtype(), Device::cpu());
			Tensor tmp_rhs(rhs.shape(), rhs.dtype(), Device::cpu());
			tmp_lhs.copyFrom(Context(), lhs);
			tmp_rhs.copyFrom(Context(), rhs);
			switch (lhs.dtype())
			{
				case DataType::FLOAT16:
					return diff_for_test_fp16(reinterpret_cast<uint16_t*>(tmp_lhs.data()), reinterpret_cast<uint16_t*>(tmp_rhs.data()),
							tmp_lhs.volume());
				case DataType::FLOAT32:
					return diff_for_test_fp32(reinterpret_cast<float*>(tmp_lhs.data()), reinterpret_cast<float*>(tmp_rhs.data()), tmp_lhs.volume());
				default:
					throw DataTypeNotSupported(METHOD_NAME, lhs.dtype());
			}
			return 0.0;
		}
		double normForTest(const Tensor &tensor)
		{
			Tensor tmp(tensor.shape(), tensor.dtype(), Device::cpu());
			tmp.copyFrom(Context(), tensor);
			switch (tmp.dtype())
			{
				case DataType::FLOAT16:
					return norm_for_test_fp16(reinterpret_cast<uint16_t*>(tmp.data()), tmp.volume());
				case DataType::FLOAT32:
					return norm_for_test_fp32(reinterpret_cast<float*>(tmp.data()), tmp.volume());
				default:
					throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
			}
			return 0.0;
		}
		double sumForTest(const Tensor &tensor)
		{
			Tensor tmp(tensor.shape(), tensor.dtype(), Device::cpu());
			tmp.copyFrom(Context(), tensor);
			switch (tmp.dtype())
			{
				case DataType::FLOAT16:
					return sum_for_test_fp16(reinterpret_cast<uint16_t*>(tmp.data()), tmp.volume());
				case DataType::FLOAT32:
					return sum_for_test_fp32(reinterpret_cast<float*>(tmp.data()), tmp.volume());
				default:
					throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
			}
			return 0.0;
		}
		void abs(Tensor &tensor)
		{
			Tensor tmp(tensor.shape(), tensor.dtype(), Device::cpu());
			tmp.copyFrom(Context(), tensor);
			switch (tmp.dtype())
			{
				case DataType::FLOAT16:
					return abs_for_test_fp16(reinterpret_cast<uint16_t*>(tmp.data()), tmp.volume());
				case DataType::FLOAT32:
					return abs_for_test_fp32(reinterpret_cast<float*>(tmp.data()), tmp.volume());
				default:
					throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
			}
			tensor.copyFrom(Context(), tmp);
		}

		bool has_device_supporting(DataType dtype)
		{
			if (Device::numberOfCudaDevices() > 0)
				return Device::cuda(0).supportsType(dtype);
			if (Device::numberOfOpenCLDevices() > 0)
				return Device::opencl(0).supportsType(dtype);
			return false;
		}
		Device get_device_for_test()
		{
			if (Device::numberOfCudaDevices() > 0)
				return Device::cuda(0);
			if (Device::numberOfOpenCLDevices() > 0)
				return Device::opencl(0);
			throw std::logic_error("No device for test");
		}

	} /* namespace testing */
} /* namespace ml */

