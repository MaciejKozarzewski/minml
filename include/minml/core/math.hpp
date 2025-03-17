/*
 * math.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_MATH_HPP_
#define MINML_MATH_HPP_

#include <cstddef>
#include <initializer_list>
#include <array>

namespace ml
{
	class Context;
	class Shape;
	class Tensor;
	class AffineTransform;
	enum class DataType;
	enum class ActivationType;
}

namespace ml
{
	void unpackInput(const Context &context, Tensor &dst, const Tensor &src);
	void convertType(const Context &context, void *dst, DataType dst_dtype, const void *src, DataType src_dtype, int elements);
	void transpose_021(const Context &context, const Tensor &input, Tensor &output);

	void winogradWeightTransform(const Context &context, const Tensor &weights, Tensor &matrices, bool invert);
	void winogradInputTransform(const Context &context, const Shape &weight_shape, const Tensor &input, Tensor &matrices);
	void winogradOutputTransform(const Context &context, const Shape &weight_shape, const Tensor &matrices, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act);
	void winogradGradientTransform(const Context &context, const Shape &weight_shape, const Tensor &gradient, Tensor &matrices);
	void winogradUpdateTransform(const Context &context, const Tensor &matrices, Tensor &update);

	void im2row(const Context &context, Tensor &output, const Tensor &input, int kernel_size, bool invert, const void *padding);
	void depthToSpace(const Context &context, const Tensor &input, Tensor &output);
	void spaceToDepth(const Context &context, const Tensor &input, Tensor &output);

	void convolutionImplicitGemmForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act);

	void depthwiseConvForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias);
	void depthwiseConvBackward(const Context &context, const Tensor &gradient_next, const Tensor &weights, Tensor &gradient_prev);
	void depthwiseConvUpdate(const Context &context, const Tensor &input, const Tensor &gradient_next, Tensor &weights_update);

	void globalAvgAndMaxPoolingForward(const Context &context, const Tensor &input, Tensor &output);
	void globalAvgAndMaxPoolingBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input,
			const Tensor &output);
	void globalBroadcastingForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &bias, ActivationType act);
	void globalBroadcastingBackward(const Context &context, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &output, ActivationType act);

	void globalAveragePoolingForward(const Context &context, const Tensor &input, Tensor &output);
	void globalAveragePoolingBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next);
	void channelScalingForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &scales);
	void channelScalingBackward(const Context &context, Tensor &gradient_prev_0, Tensor &gradient_prev_1, const Tensor &gradient_next,
			const Tensor &input_0, const Tensor &input_1);

	void gemm(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta);
	void gemmBatched(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta);

	/*
	 * Computes D = act(alpha * op_A(A) * op_B(B) + beta * C)
	 */
	void gemm_ex(const Context &context, Tensor &D, float alpha, char opA, const Tensor &A, char opB, const Tensor &B, float beta, const Tensor &C,
			const Tensor &bias, ActivationType act);

	void addBiasAct(const Context &context, Tensor &output, const Tensor &input, const Tensor &bias, ActivationType act);

	void batchnormInference(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, ActivationType act);
	void batchnormForward(const Context &context, const Tensor &input, Tensor &output, Tensor &weights, Tensor &running_stats, int running_stat_idx,
			ActivationType act);
	void batchnormBackward(const Context &context, const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next,
			const Tensor &weights, Tensor &weights_update, const Tensor &running_stats, int running_stat_idx, ActivationType act);
	void batchnormUpdate(const Context &context, const Tensor &running_stat, int stats_to_average, Tensor &weights, bool use_gamma, bool use_beta);
	void foldBatchnorm(const Context &context, Tensor &layer_weights, Tensor &layer_bias, const Tensor &batchnorm_weights);

	/*
	 * Layer normalization
	 */
	void layernormForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias, const Tensor &ext);
	void layernormBackward(const Context &context, const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			Tensor &weights_update, Tensor &bias_update);

	/*
	 * RMS normalization
	 */
	void rmsnormForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights);
	void rmsnormBackward(const Context &context, const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			Tensor &weights_update);

	/*
	 * attention
	 */
	int multiHeadAttentionGetWorkspaceSize(const Context &context, const Shape &inputShape, const Shape &weightsShape, int num_heads, bool training);
	void multiHeadAttentionForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias,
			const Tensor &mask, Tensor &workspace, Tensor &backwardData, int num_heads, bool symmetric);
	void multiHeadAttentionBackward(const Context &context, const Tensor &input, const Tensor &weights, const Tensor &bias, const Tensor &mask,
			Tensor &gradient_prev, Tensor &gradient_next, Tensor &weights_update, Tensor &bias_update, Tensor &mask_update, Tensor &workspace,
			Tensor &backwardData, int num_heads, bool symmetric);

	void windowPartitioning(const Context &context, const Tensor &input, Tensor &output, const Shape &offset);
	void windowMerging(const Context &context, const Tensor &input, Tensor &output, const Shape &offset);

	void activationForward(const Context &context, Tensor &output, const Tensor &input, ActivationType act);
	void activationBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &output, ActivationType act);
	void softmaxForward(const Context &context, Tensor &output, const Tensor &input);
	void geluBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input);

	void emulateLowPrecision(const Context &context, Tensor &dst, const Tensor &src, DataType dtype, AffineTransform transform);
	void sumOverFirstDim(const Context &context, Tensor &dst, const Tensor &src, float beta);
	void multiplyTensors(const Context &context, Tensor &dst, const Tensor &lhs, const Tensor &rhs);
	void addTensors(const Context &context, Tensor &dst, const Tensor &src1, const Tensor &src2);
	// computes dst = alpha1 * src1 + alpha2 * src2
	void addTensors(const Context &context, Tensor &dst, float alpha1, const Tensor &src1, float alpha2, const Tensor &src2);

	float meanSquaredLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask);
	void meanSquaredGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, const Tensor &mask, float weight =
			1.0f);
	float crossEntropyLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask);
	void crossEntropyGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, const Tensor &mask, float weight =
			1.0f);
	float valueHeadLoss(const Context &context, const Tensor &output, const Tensor &target);
	void valueHeadGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, float weight = 1.0f);

	void radamOptimize(const Context &context, Tensor &weight, const Tensor &update, Tensor &momentum, Tensor &variance, float learning_rate,
			float beta1, float beta2, int step, float weight_decay);

	void l2Regularization(const Context &context, Tensor &gradient, const Tensor &param, float coefficient, float offset);

	/*
	 * quantized
	 */
	void dequantize(const Context &context, const Tensor &input, Tensor &output, AffineTransform transform);
	void quantized_depthwise_conv_forward(const Context &context, const Tensor &input, const Tensor &weights, const Tensor &scales,
			const Tensor &bias, Tensor &output, AffineTransform output_transform, int padding_value);
	void quantized_scale_shift_act(const Context &context, Tensor &output, AffineTransform output_transform, const Tensor &input,
			const Tensor &scales, const Tensor &bias, ActivationType act, const Tensor &ext, AffineTransform ext_transform);
	void transpose(const Context &context, Tensor &output, const Tensor &input, std::initializer_list<int> ordering);

	std::array<int, 3> explicit_gemm_workspace(const Shape &inputShape, const Shape &outputShape, const Shape &weightShape);
	void explicit_gemm_forward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias,
			Tensor &workspace, ActivationType activation, const Tensor &add);
	void explicit_gemm_backward(const Context &context, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &output, const Tensor &weights,
			Tensor &workspace);
	void explicit_gemm_update(const Context &context, const Tensor &input, const Tensor &gradient_next, Tensor &weight_update, Tensor &workspace);

} /* namespace ml */

#endif /* MINML_MATH_HPP_ */
