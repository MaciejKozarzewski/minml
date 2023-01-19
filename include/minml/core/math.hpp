/*
 * math.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_MATH_HPP_
#define MINML_MATH_HPP_

#include <cstddef>

namespace ml
{
	class Context;
	class Shape;
	class Tensor;
	enum class DataType;
	enum class ActivationType;
}

namespace ml
{
	void unpackInput(const Context &context, Tensor &dst, const Tensor &src);
	void convertType(const Context &context, void *dst, DataType dst_dtype, const void *src, DataType src_dtype, int elements);

	void winogradWeightTransform(const Context &context, const Tensor &weights, Tensor &matrices, bool invert, bool low_precision);
	void winogradInputTransform(const Context &context, const Shape &weight_shape, const Tensor &input, Tensor &matrices);
	void winogradOutputTransform(const Context &context, const Shape &weight_shape, const Tensor &matrices, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act);
	void winogradGradientTransform(const Context &context, const Shape &weight_shape, const Tensor &gradient, Tensor &matrices);
	void winogradUpdateTransform(const Context &context, const Tensor &matrices, Tensor &update);

	void convolutionImplicitGemmForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act);

	void convolutionFusedWinogradForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act);

	void globalAvgAndMaxPoolingForward(const Context &context, const Tensor &input, Tensor &output, Tensor &max_indices);
	void globalAvgAndMaxPoolingBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &max_indices);

	void gemm(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta);
	void gemmBatched(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta);

	void addBiasAct(const Context &context, Tensor &input, const Tensor &bias, ActivationType act);

	void batchnormInference(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, ActivationType act);
	void batchnormForward(const Context &context, const Tensor &input, Tensor &output, Tensor &weights, Tensor &running_stats, int running_stat_idx,
			ActivationType act);
	void batchnormBackward(const Context &context, const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next,
			const Tensor &weights, Tensor &weights_update, const Tensor &running_stats, int running_stat_idx, ActivationType act);
	void batchnormUpdate(const Context &context, const Tensor &running_stat, int stats_to_average, Tensor &weights, bool use_gamma, bool use_beta);
	void foldBatchnorm(const Context &context, Tensor &layer_weights, Tensor &layer_bias, const Tensor &batchnorm_weights);

	void softmaxForwardInPlace(const Context &context, Tensor &input);
	void activationForwardInPlace(const Context &context, Tensor &input, ActivationType act);
	void activationBackwardInPlace(const Context &context, Tensor &gradient, const Tensor &output, ActivationType act);

	void emulateLowPrecision(const Context &context, Tensor &dst, const Tensor &src);
	void sumOverFirstDim(const Context &context, Tensor &dst, const Tensor &src, float beta);
	void addTensors(const Context &context, Tensor &dst, const Tensor &src1, const Tensor &src2);
	float crossEntropyLoss(const Context &context, const Tensor &output, const Tensor &target);
	void crossEntropyGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target);
	void adamOptimize(const Context &context, Tensor &weight, Tensor &update, Tensor &momentum, Tensor &variance, float learning_rate, float beta1,
			float beta2);
	void l2Regularization(const Context &context, Tensor &gradient, const Tensor &param, float coefficient, float offset);

} /* namespace ml */

#endif /* MINML_MATH_HPP_ */
