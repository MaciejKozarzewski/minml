/*
 * testing_util.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/testing_util.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/random.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/DataType.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/Layer.hpp>

#include <iomanip>
#include <iostream>
#include <cmath>

namespace
{
	using namespace ml;
	template<typename T>
	T square(T x) noexcept
	{
		return x * x;
	}

	void init_for_test_int32(int32_t *ptr, size_t length, float shift, float scale)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = std::sin(i / 10.0 + shift) * scale;
	}
	void init_for_test_int8(int8_t *ptr, size_t length, float shift)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = 255 * std::sin(i / 10.0 + shift) - 128;
	}
	void init_for_test_fp64(double *ptr, size_t length, float shift, float scale)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = std::sin(i / 10.0 + shift) * scale;
	}
	void init_for_test_fp32(float *ptr, size_t length, float shift, float scale)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = std::sin(i / 10.0f + shift) * scale;
	}
	void init_for_test_fp16(uint16_t *ptr, size_t length, float shift, float scale)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = convert_fp32_to_fp16(std::sin(i / 10.0f + shift) * scale);
	}

	void init_random_fp64(double *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = 2 * randDouble() - 1;
	}
	void init_random_fp32(float *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = 2 * randFloat() - 1;
	}
	void init_random_fp16(uint16_t *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = convert_fp32_to_fp16(2 * randFloat() - 1);
	}

	double diff_for_test_int32(const int32_t *ptr1, const int32_t *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(ptr1[i] - ptr2[i]);
		return result / length;
	}
	double diff_for_test_int8(const int8_t *ptr1, const int8_t *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(ptr1[i] - ptr2[i]);
		return result / length;
	}
	double diff_for_test_fp64(const double *ptr1, const double *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(ptr1[i] - ptr2[i]);
		return result / length;
	}
	double diff_for_test_fp32(const float *ptr1, const float *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(ptr1[i] - ptr2[i]);
		return result / length;
	}
	double diff_for_test_fp16(const uint16_t *ptr1, const uint16_t *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(convert_fp16_to_fp32(ptr1[i]) - convert_fp16_to_fp32(ptr2[i]));
		return result / length;
	}

	double max_abs_diff_fp64(const double *ptr1, const double *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result = std::max(result, (double) std::abs(ptr1[i] - ptr2[i]));
		return result;
	}
	double max_abs_diff_fp32(const float *ptr1, const float *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result = std::max(result, (double) std::abs(ptr1[i] - ptr2[i]));
		return result;
	}
	double max_abs_diff_fp16(const uint16_t *ptr1, const uint16_t *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result = std::max(result, (double) std::abs(convert_fp16_to_fp32(ptr1[i]) - convert_fp16_to_fp32(ptr2[i])));
		return result;
	}

	double norm_for_test_fp32(const float *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(ptr[i]);
		return result;
	}
	double norm_for_test_fp16(const uint16_t *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += std::abs(convert_fp16_to_fp32(ptr[i]));
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
			ptr[i] = std::abs(ptr[i]);
	}
	void abs_for_test_fp16(uint16_t *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			ptr[i] = std::abs(convert_fp16_to_fp32(ptr[i]));
	}

	double l2_loss_fp64(const Tensor &output, const Tensor &target)
	{
		assert(output.dtype() == DataType::FLOAT64);
		assert(target.dtype() == DataType::FLOAT64);
		assert(output.device().isCPU());
		assert(target.device().isCPU());

		double result = 0.0;
		for (int i = 0; i < output.volume(); i++)
			result += square(reinterpret_cast<const double*>(output.data())[i] - reinterpret_cast<const double*>(target.data())[i]);
		return 0.5 * result / output.volume();
	}
	void l2_grad_fp64(Tensor &gradient, const Tensor &output, const Tensor &target)
	{
		assert(gradient.dtype() == DataType::FLOAT64);
		assert(output.dtype() == DataType::FLOAT64);
		assert(target.dtype() == DataType::FLOAT64);
		assert(gradient.device().isCPU());
		assert(output.device().isCPU());
		assert(target.device().isCPU());
		for (int i = 0; i < output.volume(); i++)
			reinterpret_cast<double*>(gradient.data())[i] = (reinterpret_cast<const double*>(output.data())[i]
					- reinterpret_cast<const double*>(target.data())[i]) / output.volume();
	}
}

namespace ml
{
	namespace testing
	{
		void initForTest(Tensor &t, double shift, double scale)
		{
			if (t.isEmpty())
				return;
			Tensor tmp(t.shape(), t.dtype(), Device::cpu());
			switch (tmp.dtype())
			{
				case DataType::FLOAT16:
					init_for_test_fp16(reinterpret_cast<uint16_t*>(tmp.data()), tmp.volume(), shift, scale);
					break;
				case DataType::FLOAT32:
					init_for_test_fp32(reinterpret_cast<float*>(tmp.data()), tmp.volume(), shift, scale);
					break;
				case DataType::FLOAT64:
					init_for_test_fp64(reinterpret_cast<double*>(tmp.data()), tmp.volume(), shift, scale);
					break;
				case DataType::INT32:
					init_for_test_int32(reinterpret_cast<int32_t*>(tmp.data()), tmp.volume(), shift, scale);
					break;
				case DataType::INT8:
					init_for_test_int8(reinterpret_cast<int8_t*>(tmp.data()), tmp.volume(), shift);
					break;
				default:
					throw DataTypeNotSupported(METHOD_NAME, tmp.dtype());
			}
			t.copyFrom(Context(), tmp);
		}
		void initRandom(Tensor &t)
		{
			if (t.isEmpty())
				return;
			Tensor tmp(t.shape(), t.dtype(), Device::cpu());
			switch (tmp.dtype())
			{
				case DataType::FLOAT16:
					init_random_fp16(reinterpret_cast<uint16_t*>(tmp.data()), tmp.volume());
					break;
				case DataType::FLOAT32:
					init_random_fp32(reinterpret_cast<float*>(tmp.data()), tmp.volume());
					break;
				case DataType::FLOAT64:
					init_random_fp64(reinterpret_cast<double*>(tmp.data()), tmp.volume());
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
				case DataType::FLOAT64:
					return diff_for_test_fp64(reinterpret_cast<double*>(tmp_lhs.data()), reinterpret_cast<double*>(tmp_rhs.data()), tmp_lhs.volume());
				case DataType::INT32:
					return diff_for_test_int32(reinterpret_cast<int32_t*>(tmp_lhs.data()), reinterpret_cast<int32_t*>(tmp_rhs.data()),
							tmp_lhs.volume());
				case DataType::INT8:
					return diff_for_test_int8(reinterpret_cast<int8_t*>(tmp_lhs.data()), reinterpret_cast<int8_t*>(tmp_rhs.data()), tmp_lhs.volume());
				default:
					throw DataTypeNotSupported(METHOD_NAME, lhs.dtype());
			}
			return 0.0;
		}
		double maxAbsDiff(const Tensor &lhs, const Tensor &rhs)
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
					return max_abs_diff_fp16(reinterpret_cast<uint16_t*>(tmp_lhs.data()), reinterpret_cast<uint16_t*>(tmp_rhs.data()),
							tmp_lhs.volume());
				case DataType::FLOAT32:
					return max_abs_diff_fp32(reinterpret_cast<float*>(tmp_lhs.data()), reinterpret_cast<float*>(tmp_rhs.data()), tmp_lhs.volume());
				case DataType::FLOAT64:
					return max_abs_diff_fp64(reinterpret_cast<double*>(tmp_lhs.data()), reinterpret_cast<double*>(tmp_rhs.data()), tmp_lhs.volume());
				default:
					throw DataTypeNotSupported(METHOD_NAME, lhs.dtype());
			}
			return 0.0;
		}
		double normForTest(const Tensor &tensor)
		{
			if (tensor.isEmpty())
				return 0.0;
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
			if (tensor.isEmpty())
				return 0.0;
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
			if (tensor.isEmpty())
				return;
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

		GradientCheck::GradientCheck(const Layer &layer) :
				m_layer(std::move(layer.clone(layer.getConfig())))
		{
		}
		void GradientCheck::setInputShape(const Shape &shape)
		{
			m_layer->setInputShape(shape);
		}
		void GradientCheck::setInputShape(const std::vector<Shape> &shapes)
		{
			m_layer->setInputShape(shapes);
		}
		double GradientCheck::check(int n, double epsilon, const std::string &mode, bool verbose)
		{
			std::shared_ptr<Context> context = std::make_shared<Context>();

			m_layer->changeContext(context);
			m_layer->init();
			m_layer->convertTo(DataType::FLOAT64);

			input = std::vector<Tensor>(m_layer->numberOfInputs());
			gradient_prev = std::vector<Tensor>(m_layer->numberOfInputs());
			for (size_t i = 0; i < input.size(); i++)
			{
				input[i] = Tensor(m_layer->getInputShape(i), m_layer->dtype(), m_layer->device());
				gradient_prev[i] = zeros_like(input[i]);
			}
			output = Tensor(m_layer->getOutputShape(), m_layer->dtype(), m_layer->device());
			target = zeros_like(output);
			gradient_next = zeros_like(output);

			initRandom(target);
			for (size_t i = 0; i < input.size(); i++)
				initRandom(input[i]);

			m_layer->forward(input, output);
			l2_grad_fp64(gradient_next, output, target);
			m_layer->backward(input, output, gradient_prev, gradient_next);

			double max_diff = 0.0;
			if (mode == "input" or mode == "all")
				for (size_t i = 0; i < input.size(); i++)
				{
					for (int j = 0; j < std::min(n, input[i].volume()); j++)
					{
						const int r = (n >= input[i].volume()) ? j : randInt(input[i].volume());
						const double grad = compute_gradient(input[i], r, epsilon);
						max_diff = std::max(max_diff, std::abs(grad - reinterpret_cast<const double*>(gradient_prev[i].data())[r]));
						if (verbose)
							std::cout << r << " : " << grad << " vs " << reinterpret_cast<const double*>(gradient_prev[i].data())[r] << '\n';
					}
				}

			if (mode == "weights" or mode == "all")
				for (int j = 0; j < std::min(n, m_layer->getWeights().getParam().volume()); j++)
				{
					const int r = (n >= m_layer->getWeights().getParam().volume()) ? j : randInt(m_layer->getWeights().getParam().volume());
					const double grad = compute_gradient(m_layer->getWeights().getParam(), r, epsilon);
					max_diff = std::max(max_diff, std::abs(grad - reinterpret_cast<const double*>(m_layer->getWeights().getGradient().data())[r]));
					if (verbose)
						std::cout << r << " : " << grad << " vs " << reinterpret_cast<const double*>(m_layer->getWeights().getGradient().data())[r]
								<< '\n';
				}

			if (mode == "bias" or mode == "all")
				for (int j = 0; j < std::min(n, m_layer->getBias().getParam().volume()); j++)
				{
					const int r = (n >= m_layer->getBias().getParam().volume()) ? j : randInt(m_layer->getBias().getParam().volume());

					const double grad = compute_gradient(m_layer->getBias().getParam(), r, epsilon);
					max_diff = std::max(max_diff, std::abs(grad - reinterpret_cast<const double*>(m_layer->getBias().getGradient().data())[r]));
					if (verbose)
						std::cout << r << " : " << grad << " vs " << reinterpret_cast<const double*>(m_layer->getBias().getGradient().data())[r]
								<< '\n';
				}
			std::cout << max_diff << '\n';
			return max_diff;
		}
		/*
		 * private
		 */
		double GradientCheck::compute_gradient(Tensor &t, int idx, double epsilon)
		{
			assert(t.dtype() == DataType::FLOAT64);
			assert(t.device().isCPU());

			std::vector<double> copy(t.volume());
			t.copyToHost(copy.data(), t.sizeInBytes());

			m_layer->forward(input, output);
			const double base = l2_loss_fp64(output, target);

			reinterpret_cast<double*>(t.data())[idx] += epsilon;
			m_layer->forward(input, output);
			const double p_eps = l2_loss_fp64(output, target);

			t.copyFromHost(copy.data(), t.sizeInBytes());
			reinterpret_cast<double*>(t.data())[idx] -= epsilon;
			m_layer->forward(input, output);
			const double m_eps = l2_loss_fp64(output, target);

			t.copyFromHost(copy.data(), t.sizeInBytes());
			return (p_eps - m_eps) / (2 * epsilon);
		}

	} /* namespace testing */
} /* namespace ml */

