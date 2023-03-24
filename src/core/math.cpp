/*
 * math.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Conv2D.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>

namespace
{
	using namespace ml;

	mlDataType_t get(DataType dtype) noexcept
	{
		return static_cast<mlDataType_t>(dtype);
	}
	mlDeviceType_t get(DeviceType type) noexcept
	{
		return static_cast<mlDeviceType_t>(type);
	}
	mlActivationType_t get(ActivationType act) noexcept
	{
		return static_cast<mlActivationType_t>(act);
	}
	mlContext_t get(const Context &context) noexcept
	{
		return context.backend();
	}
	mlShape_t get(const Shape &shape) noexcept
	{
		mlShape_t result;
		result.rank = shape.rank();
		for (int i = 0; i < shape.rank(); i++)
			result.dim[i] = shape[i];
		for (int i = shape.rank(); i < 4; i++)
			result.dim[i] = 0;
		return result;
	}
	mlShape_t get_shape(const Tensor &tensor) noexcept
	{
		return get(tensor.shape());
	}
}

namespace ml
{
	void unpackInput(const Context &context, Tensor &dst, const Tensor &src)
	{
		assert(src.dtype() == DataType::INT32);
		assert(src.lastDim() == 1);
		assert(dst.dim(0) == src.dim(0));
		assert(dst.dim(1) == src.dim(1));
		assert(dst.dim(2) == src.dim(2));

		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_unpack_input(get(context), get_shape(dst), get(dst.dtype()), dst.data(), src.data());
				break;
			case DeviceType::CUDA:
				cuda_unpack_input(get(context), get_shape(dst), get(dst.dtype()), dst.data(), src.data());
				break;
		}
	}
	void convertType(const Context &context, void *dst, DataType dst_dtype, const void *src, DataType src_dtype, int elements)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_convert_type(get(context), dst, get(dst_dtype), src, get(src_dtype), elements);
				break;
			case DeviceType::CUDA:
				cuda_convert_type(get(context), dst, get(dst_dtype), src, get(src_dtype), elements);
				break;
		}
	}
	void transpose_021(const Context &context, const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 3 && output.rank() == 3);
		assert(input.dtype() == output.dtype());
		assert(input.dim(0) == output.dim(0));
		assert(input.dim(1) == output.dim(2));
		assert(input.dim(2) == output.dim(1));

		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_transpose_021(get(context), get(input.dtype()), get_shape(input), input.data(), output.data());
				break;
			case DeviceType::CUDA:
				cuda_transpose_021(get(context), get(input.dtype()), get_shape(input), input.data(), output.data());
				break;
		}
	}

	void winogradWeightTransform(const Context &context, const Tensor &weights, Tensor &matrices, bool invert, bool low_precision)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_weight_transform(get(context), get(weights.dtype()), get_shape(weights), weights.data(), matrices.data(), invert,
						low_precision);
				break;
			case DeviceType::CUDA:
				cuda_winograd_weight_transform(get(context), get(weights.dtype()), get_shape(weights), weights.data(), matrices.data(), invert,
						low_precision);
				break;
		}
	}
	void winogradInputTransform(const Context &context, const Shape &weight_shape, const Tensor &input, Tensor &matrices)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_input_transform(get(context), get(input.dtype()), get(weight_shape), get_shape(input), input.data(), matrices.data());
				break;
			case DeviceType::CUDA:
				cuda_winograd_input_transform(get(context), get(input.dtype()), get(weight_shape), get_shape(input), input.data(), matrices.data());
				break;
		}
	}
	void winogradOutputTransform(const Context &context, const Shape &weight_shape, const Tensor &matrices, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_output_transform(get(context), get(output.dtype()), get(weight_shape), get_shape(output), matrices.data(), output.data(),
						bias.data(), add.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_winograd_output_transform(get(context), get(output.dtype()), get(weight_shape), get_shape(output), matrices.data(),
						output.data(), bias.data(), add.data(), get(act));
				break;
		}
	}
	void winogradGradientTransform(const Context &context, const Shape &weight_shape, const Tensor &gradient, Tensor &matrices)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_gradient_transform(get(context), get(gradient.dtype()), get(weight_shape), get_shape(gradient), gradient.data(),
						matrices.data());
				break;
			case DeviceType::CUDA:
				cuda_winograd_gradient_transform(get(context), get(gradient.dtype()), get(weight_shape), get_shape(gradient), gradient.data(),
						matrices.data());
				break;
		}
	}
	void winogradUpdateTransform(const Context &context, const Tensor &matrices, Tensor &update)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_update_transform(get(context), get(matrices.dtype()), get_shape(update), matrices.data(), update.data());
				break;
			case DeviceType::CUDA:
				cuda_winograd_update_transform(get(context), get(matrices.dtype()), get_shape(update), matrices.data(), update.data());
				break;
		}
	}

	void convolutionImplicitGemmForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_convolution_implicit_gemm_forward(get(context), get(weights.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), output.data(), bias.data(), add.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_convolution_implicit_gemm_forward(get(context), get(weights.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), output.data(), bias.data(), add.data(), get(act));
				break;
		}
	}

	void convolutionFusedWinogradForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_convolution_fused_winograd_forward(get(context), get(weights.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), output.data(), bias.data(), add.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_convolution_fused_winograd_forward(get(context), get(weights.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), output.data(), bias.data(), add.data(), get(act));
				break;
		}
	}

	void gemm(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_gemm(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB, alpha, beta);
				break;
			case DeviceType::CUDA:
				cuda_gemm(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB, alpha,
						beta);
				break;
		}
	}
	void gemmBatched(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_gemm_batched(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB,
						alpha, beta);
				break;
			case DeviceType::CUDA:
				cuda_gemm_batched(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB,
						alpha, beta);
				break;
		}
	}

	void addBiasAct(const Context &context, Tensor &input, const Tensor &bias, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_add_bias_act(get(context), get(input.dtype()), get_shape(input), input.data(), bias.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_add_bias_act(get(context), get(input.dtype()), get_shape(input), input.data(), bias.data(), get(act));
				break;
		}
	}

	void batchnormInference(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_inference(get(context), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_inference(get(context), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
		}
	}
	void batchnormForward(const Context &context, const Tensor &input, Tensor &output, Tensor &weights, Tensor &running_stats, int running_stat_idx,
			ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_forward(get(context), get_shape(input), input.data(), output.data(), weights.data(), running_stats.data(),
						running_stat_idx, get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_forward(get(context), get_shape(input), input.data(), output.data(), weights.data(), running_stats.data(),
						running_stat_idx, get(act));
				break;
		}
	}
	void batchnormBackward(const Context &context, const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next,
			const Tensor &weights, Tensor &weights_update, const Tensor &running_stats, int running_stat_idx, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_backward(get(context), get_shape(input), input.data(), output.data(), gradient_prev.data(), gradient_next.data(),
						weights.data(), weights_update.data(), running_stats.data(), running_stat_idx, get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_backward(get(context), get_shape(input), input.data(), output.data(), gradient_prev.data(), gradient_next.data(),
						weights.data(), weights_update.data(), running_stats.data(), running_stat_idx, get(act));
				break;
		}
	}
	void batchnormUpdate(const Context &context, const Tensor &running_stat, int stats_to_average, Tensor &weights, bool use_gamma, bool use_beta)
	{
		mlShape_t shape = get_shape(running_stat);
		shape.dim[0] = stats_to_average;
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
		}
	}
	void foldBatchnorm(const Context &context, Tensor &layer_weights, Tensor &layer_bias, const Tensor &batchnorm_weights)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_fold_batchnorm(get(context), get_shape(layer_weights), layer_weights.data(), layer_bias.data(), batchnorm_weights.data());
				break;
			case DeviceType::CUDA:
				cuda_fold_batchnorm(get(context), get_shape(layer_weights), layer_weights.data(), layer_bias.data(), batchnorm_weights.data());
				break;
		}
		context.synchronize();
	}

	void activationForward(const Context &context, Tensor &output, const Tensor &input, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
		}
	}
	void activationBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &output, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(), get(act));
				break;
		}
	}

	void emulateLowPrecision(const Context &context, Tensor &dst, const Tensor &src)
	{
		if (dst.dtype() != DataType::FLOAT32 or src.dtype() != DataType::FLOAT32)
			return;
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_emulate_low_precision(get(context), get_shape(dst), dst.data(), src.data());
				break;
			case DeviceType::CUDA:
				cuda_emulate_low_precision(get(context), get_shape(dst), dst.data(), src.data());
				break;
		}
	}
	void sumOverFirstDim(const Context &context, Tensor &dst, const Tensor &src, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
			case DeviceType::CUDA:
				cuda_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
		}
	}
	void addTensors(const Context &context, Tensor &dst, const Tensor &src1, const Tensor &src2)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_add_tensors(get(context), get_shape(dst), dst.data(), src1.data(), src2.data());
				break;
			case DeviceType::CUDA:
				cuda_add_tensors(get(context), get_shape(dst), dst.data(), src1.data(), src2.data());
				break;
		}
	}
	float crossEntropyLoss(const Context &context, const Tensor &output, const Tensor &target)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::CUDA:
				return cuda_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data());
		}
		return 0.0f;
	}
	void crossEntropyGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, float weight)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::CUDA:
				cuda_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
		}
	}
	void adamOptimize(const Context &context, Tensor &weight, Tensor &update, Tensor &momentum, Tensor &variance, float learning_rate, float beta1,
			float beta2)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_adam_optimize(get(context), get_shape(weight), weight.data(), update.data(), momentum.data(), variance.data(), learning_rate,
						beta1, beta2);
				break;
			case DeviceType::CUDA:
				cuda_adam_optimize(get(context), get_shape(weight), weight.data(), update.data(), momentum.data(), variance.data(), learning_rate,
						beta1, beta2);
				break;
		}
	}
	void l2Regularization(const Context &context, Tensor &gradient, const Tensor &param, float coefficient, float offset)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_l2_regularization(get(context), get_shape(gradient), gradient.data(), param.data(), coefficient, offset);
				break;
			case DeviceType::CUDA:
				cuda_l2_regularization(get(context), get_shape(gradient), gradient.data(), param.data(), coefficient, offset);
				break;
		}
	}

}

