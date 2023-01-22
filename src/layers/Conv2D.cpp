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
	Shape get_matrices_shape(int kernel_size, const Shape &tensor_shape)
	{
		// assuming only 4x4 transform for 3x3 filter and 2x2 for 5x5 filter
		switch (kernel_size)
		{
			case 3:
				return Shape(
						{ 36, tensor_shape.firstDim() * get_tiles_count(tensor_shape[1], 4) * get_tiles_count(tensor_shape[2], 4),
								tensor_shape.lastDim() });
			case 5:
				return Shape(
						{ 36, tensor_shape.firstDim() * get_tiles_count(tensor_shape[1], 2) * get_tiles_count(tensor_shape[2], 2),
								tensor_shape.lastDim() });
		}
		return Shape();

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
		if (shapes.size() == 2 and shapes[0] != shapes[1])
			throw IllegalArgument(METHOD_NAME, "Conv2D layer expects both input shapes to be equal");

		m_input_shapes = shapes;
		m_height = shapes[0][1];
		m_width = shapes[0][2];
		m_input_filters = shapes[0][3];
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
		const bool emulate_low_precision = isTrainable() and dtype() == DataType::FLOAT32;

		choose_algorithm();
//		if (device().isCPU() and input[0].firstDim() == 1)
//		{
//			const int batch = input[0].dim(0);
//			const int height = input[0].dim(1);
//			const int width = input[0].dim(2);
//			const int filters_in = input[0].dim(3);
//			const int filters_out = output.dim(3);
//
//			const int pad = -m_kernel_size / 2;
//
//			output.zeroall(context());
//			for (int b = 0; b < batch; b++)
//				for (int out = 0; out < filters_out; out++)
//					for (int h = 0; h < height; h++)
//						for (int w = 0; w < width; w++)
//						{
//							float tmp = 0.0f;
//							if (isUsingBias())
//								tmp += getBias().getParam().get( { out });
//							for (int i = 0; i < m_kernel_size; i++)
//								for (int j = 0; j < m_kernel_size; j++)
//									if ((pad + h + i) >= 0 && (pad + h + i) < height && (pad + w + j) >= 0 && (pad + w + j) < width)
//										for (int in = 0; in < filters_in; in++)
//											tmp += getWeights().getParam().get( { out, i, j, in })
//													* input[0].get( { b, pad + h + i, pad + w + j, in });
//							output.set(tmp, { b, h, w, out });
//						}
//			activationForwardInPlace(context(), output, m_activation);
//
//			for (int out = 0; out < output.shape()[0]; out++)
//				for (int in = 0; in < output.shape()[3]; in++)
//				{
//					for (int i = 0; i < output.shape()[1]; i++)
//					{
//						for (int j = 0; j < output.shape()[2]; j++)
//							std::cout << output.get( { out, i, j, in }) << ' ';
//						std::cout << '\n';
//					}
//					std::cout << "------------------------------------\n";
//				}
//			return;
//		}

		switch (m_algorithm)
		{
			case ConvolutionAlgorithm::DIRECT:
				break;
			case ConvolutionAlgorithm::IMPLICIT_GEMM:
				break;
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

//				if (isUsingBias() and m_output_filters == 3)
//				{
//					std::cout << "------------------------------------------------\n";
//					std::cout << "input norm = " << testing::normForTest(input[0]) << '\n';
//					for (int j = 0; j < m_input_filters; j++)
//						std::cout << input[0].get( { 0, 0, 0, j }) << ' ';
//					std::cout << "weights norm = " << testing::normForTest(getWeights().getParam()) << '\n';
//					for (int i = 0; i < m_output_filters; i++)
//					{
//						for (int j = 0; j < m_input_filters; j++)
//							std::cout << getWeights().getParam().get( { i, 0, 0, j }) << ' ';
//						std::cout << '\n';
//					}
//					std::cout << "biases: " << getBias().getParam().get( { 0 }) << " " << getBias().getParam().get( { 1 }) << " "
//							<< getBias().getParam().get( { 2 }) << "\n";
//					for (int out = 0; out < output.shape()[0]; out++)
//						for (int in = 0; in < output.shape()[3]; in++)
//						{
//							for (int i = 0; i < output.shape()[1]; i++)
//							{
//								for (int j = 0; j < output.shape()[2]; j++)
//									std::cout << output.get( { out, i, j, in }) << ' ';
//								std::cout << '\n';
//							}
//							std::cout << "------------------------------------\n";
//						}
//				}

				if (isUsingBias())
					addBiasAct(context(), output, getBias().getParam(), m_activation);
				else
					activationForward(context(), output, output, m_activation);

//				std::cout << "Conv2D 1x1 " << input[0].shape() << " : " << testing::normForTest(input[0]) << " -> " << output.shape() << " : "
//						<< testing::normForTest(output) << '\n';

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
		const bool emulate_low_precision = isTrainable() and dtype() == DataType::FLOAT32;
		choose_algorithm();
//		if (device().isCPU())
//		{
//			const int batch = output.dim(0);
//			const int height = output.dim(1);
//			const int width = output.dim(2);
//			const int filters_in = gradient_prev[0].dim(3);
//			const int filters_out = gradient_next.dim(3);
//
//			const int pad = -m_kernel_size / 2;
//
//			activationBackwardInPlace(context(), gradient_next, output, m_activation);
//			gradient_prev[0].zeroall(context());
//			for (int b = 0; b < batch; b++)
//				for (int out = 0; out < filters_out; out++)
//				{
//					for (int h = 0; h < height; h++)
//						for (int w = 0; w < width; w++)
//							for (int i = 0; i < m_kernel_size; i++)
//								for (int j = 0; j < m_kernel_size; j++)
//									if ((pad + h + i) >= 0 && (pad + h + i) < height && (pad + w + j) >= 0 && (pad + w + j) < width)
//										for (int in = 0; in < filters_in; in++)
//										{
//											const float grad = gradient_next.get( { b, h, w, out });
//											const float we = getWeights().getParam().get( { out, i, j, in });
//											const float pr = gradient_prev[0].get( { b, pad + h + i, pad + w + j, in });
//											gradient_prev[0].set(pr + grad * we, { b, pad + h + i, pad + w + j, in });
//										}
//					for (int in = 0; in < filters_in; in++)
//						for (int i = 0; i < m_kernel_size; i++)
//							for (int j = 0; j < m_kernel_size; j++)
//							{
//								float tmp = getWeights().getGradient().get( { out, i, j, in });
//								for (int h = 0; h < height; h++)
//									for (int w = 0; w < width; w++)
//										if ((pad + h + i) >= 0 && (pad + h + i) < height && (pad + w + j) >= 0 && (pad + w + j) < width)
//											tmp += gradient_next.get( { b, h, w, out }) * input[0].get( { b, pad + h + i, pad + w + j, in });
//								getWeights().getGradient().set(tmp, { out, i, j, in });
//							}
//				}
//			if (isUsingBias())
//				sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 1.0f);
//			return;
//		}

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
				gemm(context(), 't', 'n', weight_update_matrix, gradient_matrix, input_matrix, 1, 1);

//				std::cout << "Conv2D 1x1 " << input[0].shape() << " -> " << output.shape() << " gradient per weight = "
//						<< testing::normForTest(weight_update_matrix) / weight_update_matrix.volume() << ", gradient sum = "
//						<< testing::sumForTest(weight_update_matrix) << ", weight norm  = " << testing::normForTest(getWeights().getParam()) << '\n';
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
			sumOverFirstDim(context(), getBias().getGradient(), gradient_next, 1.0f);
		if (gradient_prev.size() == 2)
			gradient_prev[1].copyFrom(context(), gradient_next);
	}

	void Conv2D::choose_algorithm()
	{
		if (m_kernel_size == 1)
			m_algorithm = ConvolutionAlgorithm::EXPLICIT_GEMM;
		else
			m_algorithm = ConvolutionAlgorithm::WINOGRAD_NON_FUSED;
	}

} /* namespace ml */

