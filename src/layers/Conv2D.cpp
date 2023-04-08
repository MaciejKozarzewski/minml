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
#include <minml/utils/testing_util.hpp>
#include <minml/utils/string_util.hpp>

namespace
{
	using namespace ml;
	Shape get_weight_matrices_shape(const Shape &weight_shape)
	{
		assert(weight_shape[1] == weight_shape[2]); // only square kernels
		// assuming only 4x4 transform for 3x3 filter and 2x2 for 5x5 filter
		return Shape( { 36, weight_shape.firstDim(), weight_shape.lastDim() });
	}
	int get_tiles_count(int dim, int tile_size)
	{
		return (dim + tile_size - 1) / tile_size;
	}
	Shape get_matrices_shape(int kernel_size, const Shape &shape)
	{
		// assuming only 4x4 transform for 3x3 filter and 2x2 for 5x5 filter
		switch (kernel_size)
		{
			default:
				return Shape();
			case 3:
				return Shape( { 36, shape.firstDim() * get_tiles_count(shape[1], 4) * get_tiles_count(shape[2], 4), shape.lastDim() });
			case 5:
				return Shape( { 36, shape.firstDim() * get_tiles_count(shape[1], 2) * get_tiles_count(shape[2], 2), shape.lastDim() });
		}
	}
}

namespace ml
{
	Conv2D::Conv2D(int filters, int kernelSize, const std::string &activation, bool useBias) :
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
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}

	int Conv2D::getWorkspaceSize() const noexcept
	{
		if (m_kernel_size == 1)
			return 0;
		else
		{
			const Shape tmp1 = get_matrices_shape(m_kernel_size, getInputShape());
			const Shape tmp2 = get_matrices_shape(m_kernel_size, getOutputShape());
			return tmp1.volume() + tmp2.volume() + 1024;
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
		const bool emulate_low_precision = false; // isTrainable() and dtype() == DataType::FLOAT32;

		choose_algorithm();

		switch (m_algorithm)
		{
			case ConvolutionAlgorithm::DIRECT:
				break;
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
				assert(m_kernel_size == 1);
				Tensor input_matrix = input[0].view( { input[0].shape().volumeWithoutLastDim(), input[0].lastDim() });
				Tensor output_matrix = output.view( { output.shape().volumeWithoutLastDim(), output.lastDim() });
				Tensor weight_matrix = getWeights().getParam().view( { getWeightShape().firstDim(), getWeightShape().volumeWithoutFirstDim() });

				if (input.size() == 2)
				{
					output.copyFrom(context(), input[1]);
					gemm(context(), 'n', 't', output_matrix, input_matrix, weight_matrix, 1, 1);
				}
				else
					gemm(context(), 'n', 't', output_matrix, input_matrix, weight_matrix, 1, 0);

				if (isUsingBias())
					addBiasAct(context(), output, getBias().getParam(), m_activation);
				else
					activationForward(context(), output, output, m_activation);

				break;
			}
			case ConvolutionAlgorithm::WINOGRAD_NON_FUSED:
			{
				if (m_transformed_weights == nullptr)
					m_transformed_weights = std::make_unique<Tensor>(get_weight_matrices_shape(getWeightShape()), dtype(), device());
				if (m_are_weights_transformed == false)
				{
					m_are_weights_transformed = true;
					winogradWeightTransform(context(), getWeights().getParam(), *m_transformed_weights, false, emulate_low_precision);
				}
				Tensor input_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, input[0].shape()));
				Tensor output_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, output.shape()), input_matrices.volume());

				winogradInputTransform(context(), getWeightShape(), input[0], input_matrices);
				gemmBatched(context(), 'n', 't', output_matrices, input_matrices, *m_transformed_weights, 1, 0);
				if (input.size() == 1)
					winogradOutputTransform(context(), getWeightShape(), output_matrices, output, getBias().getParam(),
							Tensor(Shape(), dtype(), device()), m_activation);
				else
					winogradOutputTransform(context(), getWeightShape(), output_matrices, output, getBias().getParam(), input[1], m_activation);
				break;
			}
			case ConvolutionAlgorithm::WINOGRAD_FUSED:
			{
				break;
			}
		}
	}
	void Conv2D::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		assert(input.size() == 1 && gradient_prev.size() == 1);
		const bool emulate_low_precision = false; // isTrainable() and dtype() == DataType::FLOAT32;
		choose_algorithm();

		activationBackward(context(), gradient_next, gradient_next, output, m_activation);
		switch (m_algorithm)
		{
			default:
				throw NotImplemented(METHOD_NAME, "");
			case ConvolutionAlgorithm::EXPLICIT_GEMM:
			{
				assert(m_kernel_size == 1);

				Tensor gradient_prev_matrix = gradient_prev[0].view( { gradient_prev[0].shape().volumeWithoutLastDim(), gradient_prev[0].lastDim() });
				Tensor weight_matrix = getWeights().getParam().view( { getWeightShape().firstDim(), getWeightShape().volumeWithoutFirstDim() });
				Tensor gradient_next_matrix = gradient_next.view(
						{ gradient_next.shape().volumeWithoutLastDim(), getWeightShape().volumeWithoutLastDim() });
				gemm(context(), 'n', 'n', gradient_prev_matrix, gradient_next_matrix, weight_matrix, 1, 0);

				Tensor input_matrix = input[0].view( { input[0].shape().volumeWithoutLastDim(), input[0].lastDim() });
				Tensor weight_update_matrix = getWeights().getGradient().view(
						{ getWeightShape().firstDim(), getWeightShape().volumeWithoutFirstDim() });
				Tensor gradient_matrix = gradient_next.view( { gradient_next.shape().volumeWithoutLastDim(), gradient_next.lastDim() });
				gemm(context(), 't', 'n', weight_update_matrix, gradient_matrix, input_matrix, 1, 0);

				break;
			}
			case ConvolutionAlgorithm::WINOGRAD_NON_FUSED:
			{
				if (m_transformed_weights == nullptr)
					m_transformed_weights = std::make_unique<Tensor>(get_weight_matrices_shape(getWeightShape()), dtype(), device());

				m_are_weights_transformed = false;
				winogradWeightTransform(context(), getWeights().getParam(), *m_transformed_weights, true, emulate_low_precision);

				Tensor gradient_next_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, output.shape()));
				Tensor gradient_prev_matrices = m_workspace.lock()->view(get_matrices_shape(m_kernel_size, input[0].shape()),
						gradient_next_matrices.volume());

				winogradInputTransform(context(), getWeightShape(), gradient_next, gradient_next_matrices);
				gemmBatched(context(), 'n', 'n', gradient_prev_matrices, gradient_next_matrices, *m_transformed_weights, 1, 0);
				winogradOutputTransform(context(), getWeightShape(), gradient_prev_matrices, gradient_prev[0], Tensor( { }, dtype(), device()),
						Tensor( { }, dtype(), device()), ActivationType::LINEAR);

				winogradGradientTransform(context(), getWeightShape(), gradient_next, gradient_next_matrices);
				winogradInputTransform(context(), getWeightShape(), input[0], gradient_prev_matrices);
				gemmBatched(context(), 't', 'n', *m_transformed_weights, gradient_next_matrices, gradient_prev_matrices, 1, 0);
				winogradUpdateTransform(context(), *m_transformed_weights, getWeights().getGradient());
				break;
			}
		}
		if (isUsingBias())
			sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 0);
	}

	void Conv2D::choose_algorithm()
	{
		switch (m_kernel_size)
		{
			case 1:
			{
#ifdef USE_CUDNN
				if (not isTrainable() and startsWith(context().device().info(), "NVIDIA RTX") and dtype() == DataType::FLOAT16)
					m_algorithm = ConvolutionAlgorithm::IMPLICIT_GEMM;
				else
					m_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
#else
				m_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
#endif
				break;
			}
			case 3:
			case 5:
			{
#ifdef USE_CUDNN
				if (not isTrainable() and startsWith(context().device().info(), "NVIDIA RTX"))
					m_algorithm = ConvolutionAlgorithm::IMPLICIT_GEMM;
				else
					m_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
#else
				m_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
#endif
				break;
			}
			default:
				break;
		}
	}

} /* namespace ml */

