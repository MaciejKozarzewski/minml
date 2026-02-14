/*
 * Conv2D.cpp
 *
 *  Created on: Feb 26, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/Conv2D.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/string_util.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/testing_util.hpp>

#include <cmath>

namespace
{
	using namespace ml;
	int square(int i) noexcept
	{
		return i * i;
	}

	Shape get_weight_matrices_shape(const Shape &weight_shape, int tile_size) noexcept
	{
		assert(weight_shape[1] == weight_shape[2]); // only square kernels
		return Shape( { square(tile_size + weight_shape[1] - 1), weight_shape.firstDim(), weight_shape.lastDim() });
	}
	int get_tiles_count(int dim, int tile_size) noexcept
	{
		assert(tile_size > 0);
		return (dim + tile_size - 1) / tile_size;
	}
	Shape get_matrices_shape(int kernel_size, int tile_size, const Shape &shape) noexcept
	{
		return Shape( { square(tile_size + kernel_size - 1), shape[0] * get_tiles_count(shape[1], tile_size) * get_tiles_count(shape[2], tile_size),
				shape[3] });
	}
	std::string to_string(const AffineTransform &t)
	{
		return std::to_string(t.scale()) + " * x + " + std::to_string(t.shift());
	}

}

namespace ml
{
	Conv2D::Conv2D(int filters, int kernelSize, std::string activation, bool useBias) :
			Layer(activation)
	{
		m_output_filters = filters;
		m_kernel_size = kernelSize;
		m_use_bias = useBias;
	}

	Conv2D& Conv2D::useBias(bool b) noexcept
	{
		if (b != m_use_bias)
			m_bias = nullptr;
		m_use_bias = b;
		return *this;
	}
	bool Conv2D::isUsingBias() const noexcept
	{
		return m_use_bias;
	}

	void Conv2D::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1 and shapes.size() != 2)
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects either one or two input shapes");
		if (shapes[0].rank() != 4)
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects 4D shapes");

		m_input_shapes = shapes;
		m_height = shapes[0][1];
		m_width = shapes[0][2];
		m_input_filters = shapes[0][3];

		if (shapes.size() == 2 and shapes[1] != getOutputShape())
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects second input shape to be equal to the output shape");
	}
	Shape Conv2D::getOutputShape() const
	{
		return Shape( { getInputShape().firstDim(), m_height, m_width, m_output_filters });
	}
	Shape Conv2D::getWeightShape() const
	{
		return Shape( { m_output_filters, m_kernel_size, m_kernel_size, m_input_filters });
	}
	Shape Conv2D::getBiasShape() const
	{
		if (m_use_bias)
			return Shape( { m_output_filters });
		else
			return Shape();
	}

	std::string Conv2D::name() const
	{
		return "Conv2D";
	}
	Json Conv2D::getConfig() const
	{
		Json result = Layer::getConfig();
		result["input_filters"] = m_input_filters;
		result["output_filters"] = m_output_filters;
		result["kernel_size"] = m_kernel_size;
		result["use_bias"] = m_use_bias;
		return result;
	}

	std::unique_ptr<Layer> Conv2D::clone(const Json &config) const
	{
		std::unique_ptr<Conv2D> result = std::make_unique<Conv2D>(config["output_filters"], config["kernel_size"], config["nonlinearity"],
				config["use_bias"]);
		result->m_input_filters = config["input_filters"];
		result->loadConfig(config);
		return result;
	}
	void Conv2D::init()
	{
		m_initializer.init_weights(context(), getWeights(), 0.1f * std::sqrt(2.0f / (m_input_filters + m_output_filters)), 0.0f);
		m_initializer.init_bias(context(), getBias(), 0.1f, 0.0f);
	}

	int Conv2D::getWorkspaceSize() const noexcept
	{
		if (dtype() == DataType::INT8)
		{
			const int tmp_output_size = getOutputShape().volume() * sizeOf(DataType::INT32);
			if (m_kernel_size == 1)
				return tmp_output_size;
			else
			{
				const int tmp_input_size = (getInputShape().volume() * square(m_kernel_size) * sizeOf(DataType::INT8) + 3) / 4;
				return tmp_output_size + tmp_input_size;
			}
		}

		if (m_kernel_size == 1)
			return 0;
		else
		{
			choose_algorithm();

			const std::array<int, 3> gemm1 = explicit_gemm_workspace(getInputShape(), getOutputShape(), getWeightShape());

			const Shape win1 = get_matrices_shape(m_kernel_size, m_winograd_tile_size, getInputShape());
			const Shape win2 = get_matrices_shape(m_kernel_size, m_winograd_tile_size, getOutputShape());
			const Shape win3 = get_weight_matrices_shape(getWeightShape(), m_winograd_tile_size);
			int result = 0;
			switch (m_forward_algorithm)
			{
				case ConvolutionAlgorithm::EXPLICIT_GEMM:
					result = std::max(result, gemm1[0]);
					break;
				case ConvolutionAlgorithm::WINOGRAD_NON_FUSED:
					result = std::max(result, win1.volume() + win2.volume() + 1024);
					break;
				default:
					break;
			}
			switch (m_backward_algorithm)
			{
				case ConvolutionAlgorithm::EXPLICIT_GEMM:
					result = std::max(result, std::max(gemm1[1], gemm1[2]));
					break;
				case ConvolutionAlgorithm::WINOGRAD_NON_FUSED:
					result = std::max(result, win1.volume() + win2.volume() + 2 * win3.volume() + 1024);
					break;
				default:
					break;
			}
			return result;
		}
	}
	void Conv2D::changeContext(std::shared_ptr<Context> &context)
	{
		Layer::changeContext(context);
		if (m_transformed_weights != nullptr)
			m_transformed_weights->moveTo(device());
	}
	void Conv2D::invalidateWeightsCache()
	{
		m_transformed_weights = nullptr;
		m_are_weights_transformed = false;
	}
	void Conv2D::convertTo(DataType newType)
	{
		Layer::convertTo(newType);
		invalidateWeightsCache();
	}

	void Conv2D::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		if (isInteger(input[0].dtype()))
		{
			assert(output.dtype() == dtype());
			if (device().isCUDA())
			{
				Tensor input_matrix;
				Tensor output_matrix = m_workspace.lock()->view( { output.shape().volumeWithoutLastDim(), output.lastDim() });
				Tensor weight_matrix = getWeights().getParam().view( { getWeightShape().firstDim(), getWeightShape().volumeWithoutFirstDim() });
				if (m_kernel_size == 1)
					input_matrix = input[0].view( { input[0].shape().volumeWithoutLastDim(), input[0].lastDim() });
				else
				{
					input_matrix = m_workspace.lock()->view( { input[0].shape().volumeWithoutLastDim(), weight_matrix.lastDim() },
							output_matrix.volume());
					input_matrix.reinterpretAs(DataType::INT8);

					const int8_t input_padding = m_input_transforms.at(0).get_inverse()(0.0f);
					im2row(context(), input_matrix, input[0], m_kernel_size, false, &input_padding);
				}
				output_matrix.reinterpretAs(DataType::INT32);

				gemm(context(), 'n', 't', output_matrix, input_matrix, weight_matrix, 1, 0);
				if (input.size() == 1)
					quantized_scale_shift_act(context(), output, m_output_transform, output_matrix, m_channel_scales, getBias().getParam(),
							m_activation, Tensor(), AffineTransform());
				else
					quantized_scale_shift_act(context(), output, m_output_transform, output_matrix, m_channel_scales, getBias().getParam(),
							m_activation, input[1], m_input_transforms[1]);
			}
			if (device().isCPU())
			{
				const int pad_h = -(m_kernel_size - 1) / 2;
				const int pad_w = -(m_kernel_size - 1) / 2;

				const int32_t input_zero = get_zero<int32_t>(m_input_transforms[0]);
				const AffineTransform output_to_int = m_output_transform.get_inverse();

//				std::cout << input[0].info() << '\n';
//				std::cout << "m_input_transforms[0] " << to_string(m_input_transforms[0]) << '\n';
//				std::cout << "m_output_transform " << to_string(m_output_transform) << '\n';
//				std::cout << "output_to_int8 " << to_string(output_to_int8) << '\n';
//				std::cout << "channel scale = " << (float) m_channel_scales.at( { 10 }) << '\n';
//				std::cout << "bias = " << (float) getBias().getParam().at( { 10 }) << '\n';

				for (int b = 0; b < input[0].dim(0); b++)
					for (int h = 0; h < input[0].dim(1); h++)
						for (int w = 0; w < input[0].dim(2); w++)
							for (int out = 0; out < output.dim(3); out++)
							{
								int32_t acc = 0;
								for (int i = 0; i < m_kernel_size; i++)
									for (int j = 0; j < m_kernel_size; j++)
									{
										const int x = pad_h + h + i;
										const int y = pad_w + w + j;
										if (0 <= x and x < input[0].dim(1) and 0 <= y and y < input[0].dim(2))
										{
											for (int in = 0; in < input[0].dim(3); in++)
												acc += (int) getWeights().getParam().at( { out, i, j, in }) * (int) input[0].at( { b, x, y, in });
										}
										else
										{
											for (int in = 0; in < input[0].dim(3); in++)
												acc += (int) getWeights().getParam().at( { out, i, j, in }) * input_zero;
										}
									}
								// quantization shift of the input tensor is absorbed into the bias
								float tmp = static_cast<float>(acc) * (float) m_channel_scales.at( { out })
										+ (float) getBias().getParam().at( { out });

								if (input.size() == 2)
									tmp += m_input_transforms[1]((float) input[1].at( { b, h, w, out }));

								switch (m_activation)
								{
									default:
									case ActivationType::LINEAR:
										break;
									case ActivationType::SIGMOID:
										tmp = 1.0f / (1.0f + std::exp(-tmp));
										break;
									case ActivationType::TANH:
										tmp = std::tanh(tmp);
										break;
									case ActivationType::RELU:
										tmp = std::max(0.0f, tmp);
										break;
									case ActivationType::EXP:
										tmp = std::exp(tmp);
										break;
								}

								if (isInteger(output.dtype()))
									output.at( { b, h, w, out }) = quantize(output_to_int(tmp), m_quantization_bits);
								if (output.dtype() == DataType::FLOAT32)
									output.at( { b, h, w, out }) = tmp;
							}
			}
			return;
		}

		choose_algorithm();

		switch (m_forward_algorithm)
		{
			case ConvolutionAlgorithm::IMPLICIT_GEMM:
			{
				if (device().isCUDA())
				{
					if (input.size() == 2)
						convolutionImplicitGemmForward(context(), input[0], getWeights().getParam(), output, getBias().getParam(), input[1],
								m_activation);
					else
						convolutionImplicitGemmForward(context(), input[0], getWeights().getParam(), output, getBias().getParam(), Tensor(),
								m_activation);
				}
				break;
			}
			case ConvolutionAlgorithm::EXPLICIT_GEMM:
			{
				if (input.size() == 2)
					explicit_gemm_forward(context(), input[0], output, getWeights().getParam(), getBias().getParam(), *m_workspace.lock(),
							m_activation, input[1]);
				else
					explicit_gemm_forward(context(), input[0], output, getWeights().getParam(), getBias().getParam(), *m_workspace.lock(),
							m_activation, Tensor());
				break;
			}
			case ConvolutionAlgorithm::WINOGRAD_NON_FUSED:
			{
				if (m_transformed_weights == nullptr)
					m_transformed_weights = std::make_unique<Tensor>(get_weight_matrices_shape(getWeightShape(), m_winograd_tile_size), dtype(),
							device());
				if (m_are_weights_transformed == false)
				{
					m_are_weights_transformed = true;
					winogradWeightTransform(context(), getWeights().getParam(), *m_transformed_weights, false);
				}

				Tensor input_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, m_winograd_tile_size, input[0].shape()));
				Tensor output_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, m_winograd_tile_size, output.shape()),
						input_matrices.volume());

				winogradInputTransform(context(), getWeightShape(), input[0], input_matrices);

				gemmBatched(context(), 'n', 't', output_matrices, input_matrices, *m_transformed_weights, 1, 0);

				if (input.size() == 1)
					winogradOutputTransform(context(), getWeightShape(), output_matrices, output, getBias().getParam(),
							Tensor(Shape(), dtype(), device()), m_activation, 0.0f);
				else
					winogradOutputTransform(context(), getWeightShape(), output_matrices, output, getBias().getParam(), input[1], m_activation, 0.0f);

				break;
			}
		}
	}
	void Conv2D::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		choose_algorithm();

		Tensor flattened_dy = gradient_next.view().flatten( { 0, 1, 2 });
		Tensor flattened_dx = gradient_prev[1].view().flatten( { 0, 1, 2 });
		const Tensor flattened_y = output.view().flatten( { 0, 1, 2 });
		if (gradient_prev.size() == 1)
		{
			Tensor empty;
			fusedBiasActCopyBackward(context(), flattened_dy, flattened_y, 0.0f, empty, 0.0f, getBias().getGradient(), m_activation);
		}
		else
			fusedBiasActCopyBackward(context(), flattened_dy, flattened_y, beta[1], flattened_dx, 0.0f, getBias().getGradient(), m_activation);

		switch (m_backward_algorithm)
		{
			default:
				throw NotImplemented(METHOD_NAME, "");
			case ConvolutionAlgorithm::EXPLICIT_GEMM:
			{
				explicit_gemm_backward(context(), gradient_prev[0], gradient_next, output, getWeights().getParam(), *m_workspace.lock(), beta[0]);
				explicit_gemm_update(context(), input[0], gradient_next, getWeights().getGradient(), *m_workspace.lock());
				break;
			}
			case ConvolutionAlgorithm::WINOGRAD_NON_FUSED:
			{
				if (m_transformed_weights == nullptr)
					m_transformed_weights = std::make_unique<Tensor>(get_weight_matrices_shape(getWeightShape(), m_winograd_tile_size), dtype(),
							device());

				m_are_weights_transformed = false;
				winogradWeightTransform(context(), getWeights().getParam(), *m_transformed_weights, true);

				Tensor gradient_next_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, m_winograd_tile_size, output.shape()));
				Tensor gradient_prev_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, m_winograd_tile_size, input[0].shape()),
						gradient_next_matrices.volume());
				Tensor weight_update_matrices = m_workspace.lock()->view(get_weight_matrices_shape(getWeightShape(), m_winograd_tile_size),
						gradient_next_matrices.volume() + gradient_prev_matrices.volume());
				weight_update_matrices.reinterpretAs(DataType::FLOAT32);

				winogradInputTransform(context(), getWeightShape(), gradient_next, gradient_next_matrices);
				gemmBatched(context(), 'n', 'n', gradient_prev_matrices, gradient_next_matrices, *m_transformed_weights, 1, 0);
				winogradOutputTransform(context(), getWeightShape(), gradient_prev_matrices, gradient_prev[0], Tensor(), Tensor(),
						ActivationType::LINEAR, beta[0]);

				winogradGradientTransform(context(), getWeightShape(), gradient_next, gradient_next_matrices);
				winogradInputTransform(context(), getWeightShape(), input[0], gradient_prev_matrices);
				gemmBatched(context(), 't', 'n', weight_update_matrices, gradient_next_matrices, gradient_prev_matrices, 1, 0);
				winogradUpdateTransform(context(), weight_update_matrices, getWeights().getGradient());
				break;
			}
		}
	}

	void Conv2D::choose_algorithm() const
	{

		switch (device().type())
		{
			case DeviceType::CPU:
			{
				switch (m_kernel_size)
				{
					case 1:
						m_forward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
						m_backward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
						break;
					case 3:
						m_forward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_backward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_winograd_tile_size = isTrainable() ? 4 : 5;
						break;
					case 5:
						m_forward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
						m_backward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_winograd_tile_size = 2;
						break;
				}
				break;
			}
			case DeviceType::CUDA:
			{
#ifdef USE_CUDNN
				const bool can_use_cudnn = startsWith(context().device().info(), "NVIDIA GeForce RTX") and dtype() == DataType::FLOAT16 and not isTrainable()
					and m_activation == ActivationType::RELU;
#else
				const bool can_use_cudnn = false;
#endif
				if (can_use_cudnn)
					m_forward_algorithm = ConvolutionAlgorithm::IMPLICIT_GEMM;
				else
				{
					switch (m_kernel_size)
					{
						case 1:
							m_forward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
							m_backward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
							break;
						case 3:
							m_forward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
							m_backward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
							m_winograd_tile_size = 4;
							break;
						case 5:
							m_forward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
							m_backward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
							m_winograd_tile_size = 2;
							break;
					}
				}
				break;
			}
			case DeviceType::OPENCL:
			{
				switch (m_kernel_size)
				{
					case 1:
						m_forward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
						m_backward_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
						break;
					case 3:
						m_forward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_backward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_winograd_tile_size = 4;
						break;
					case 5:
						m_forward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_backward_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
						m_winograd_tile_size = 2;
						break;
				}
				break;
			}
		}
	}

} /* namespace ml */

