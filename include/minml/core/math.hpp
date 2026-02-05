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
#include <vector>
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
	void convertTensor(const Context &context, Tensor &dst, const Tensor &src);
	void convertType(const Context &context, void *dst, DataType dst_dtype, const void *src, DataType src_dtype, int elements);
	void transpose_021(const Context &context, const Tensor &input, Tensor &output);

	void winogradWeightTransform(const Context &context, const Tensor &weights, Tensor &matrices, bool invert);
	void winogradInputTransform(const Context &context, const Shape &weight_shape, const Tensor &input, Tensor &matrices);
	void winogradOutputTransform(const Context &context, const Shape &weight_shape, const Tensor &matrices, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act, float beta);
	void winogradGradientTransform(const Context &context, const Shape &weight_shape, const Tensor &gradient, Tensor &matrices);
	void winogradUpdateTransform(const Context &context, const Tensor &matrices, Tensor &update);

	void im2row(const Context &context, Tensor &output, const Tensor &input, int kernel_size, bool invert, const void *padding);
	void depthToSpace(const Context &context, const Tensor &input, Tensor &output, float beta);
	void spaceToDepth(const Context &context, const Tensor &input, Tensor &output, float beta);

	void convolutionImplicitGemmForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act);

	void fusedConvBlockForward(const Context &context, const Tensor &input, const Tensor &dwconv_weights, const Tensor &dwconv_bias,
			const Tensor &first_conv_weights, const Tensor &first_conv_bias, const Tensor &second_conv_weights, const Tensor &second_conv_bias,
			Tensor &output);

	void depthwiseConvForward(const Context &context, float alpha, const Tensor &input, const Tensor &weights, float beta, Tensor &output,
			const Tensor &bias);
	void depthwiseConvBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &weights, float beta,
			Tensor &gradient_prev);
	void depthwiseConvUpdate(const Context &context, float alpha, const Tensor &input, const Tensor &gradient_next, float beta,
			Tensor &weights_update);

	void averagePoolingForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output, int size);
	void averagePoolingBackward(const Context &context, float alpha, const Tensor &gradient_next, float beta, Tensor &gradient_prev, int size);

	void globalAveragePoolingForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output);
	void globalAveragePoolingBackward(const Context &context, float alpha, const Tensor &gradient_next, float beta, Tensor &gradient_prev);
	void channelScalingForward(const Context &context, float alpha, const Tensor &input, const Tensor &scales, float beta, Tensor &output);
	void channelScalingBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &input, const Tensor &scales,
			float beta_input, Tensor &gradient_prev, float beta_scales, Tensor &gradient_scales);
	void channelAveragePoolingForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output);
	void channelAveragePoolingBackward(const Context &context, float alpha, const Tensor &gradient_next, float beta, Tensor &gradient_prev);
	void spatialScalingForward(const Context &context, float alpha, const Tensor &input, const Tensor &scales, float beta, Tensor &output);
	void spatialScalingBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &input, const Tensor &scales,
			float beta_input, Tensor &gradient_prev, float beta_scales, Tensor &gradient_scales);

	void gemm(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta);
	void gemmBatched(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta);

	/*
	 * Computes D = act(alpha * op_A(A) * op_B(B) + beta * C)
	 */
	void gemm_ex(const Context &context, Tensor &D, float alpha, char opA, const Tensor &A, char opB, const Tensor &B, float beta, const Tensor &C,
			const Tensor &bias, ActivationType act);

	void addBiasAct(const Context &context, float alpha, const Tensor &input, const Tensor &bias, float beta, Tensor &output, ActivationType act);

	void batchnormInference(const Context &context, float alpha, const Tensor &input, const Tensor &weights, const Tensor &bias,
			const Tensor &avg_var, float beta, Tensor &output, ActivationType act);
	void batchnormForward(const Context &context, float alpha, const Tensor &input, const Tensor &weights, const Tensor &bias, float beta,
			Tensor &output, Tensor &running_stats, ActivationType act);
	void batchnormBackward(const Context &context, float alpha, const Tensor &input, const Tensor &output, Tensor &gradient_next,
			const Tensor &weights, const Tensor &bias, float beta_prev, Tensor &gradient_prev, float beta_update, Tensor &weights_update,
			Tensor &bias_update, const Tensor &running_stats, ActivationType act);
	void batchnormUpdate(const Context &context, const Tensor &running_stats, Tensor &avg_var);
	void foldBatchnorm(const Context &context, Tensor &layer_weights, Tensor &layer_bias, const Tensor &bn_weights, const Tensor &bn_bias,
			const Tensor &bn_avg_var);

	/*
	 * Layer normalization
	 */
	void layernormForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias, const Tensor &ext);
	void layernormBackward(const Context &context, const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			Tensor &weights_update, Tensor &bias_update, float beta);

	/*
	 * RMS normalization
	 */
	void rmsnormForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights);
	void rmsnormBackward(const Context &context, const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			Tensor &weights_update, float beta);

	/*
	 * attention
	 */
	int multiHeadAttentionGetWorkspaceSize(const Context &context, const Shape &inputShape, const Shape &weightsShape, int num_heads, bool training);
	void multiHeadAttentionForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias,
			const Tensor &mask, Tensor &workspace, Tensor &backwardData, int num_heads, bool symmetric);
	void multiHeadAttentionBackward(const Context &context, const Tensor &input, const Tensor &weights, const Tensor &bias, const Tensor &mask,
			Tensor &gradient_prev, Tensor &gradient_next, Tensor &weights_update, Tensor &bias_update, Tensor &mask_update, Tensor &workspace,
			Tensor &backwardData, int num_heads, bool symmetric, float beta);

	void windowPartitioning(const Context &context, const Tensor &input, Tensor &output, const Shape &offset);
	void windowMerging(const Context &context, const Tensor &input, Tensor &output, const Shape &offset);

	/*
	 * activations
	 */
	void activationForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output, ActivationType act);
	void activationBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &output, float beta, Tensor &gradient_prev,
			ActivationType act);
	void softmaxForward(const Context &context, Tensor &output, const Tensor &input);
	void softmaxBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &output, float beta, Tensor &gradient_prev);
	void fusedBiasActCopyBackward(const Context &context, Tensor &gradient_next, const Tensor &output, float beta_prev, Tensor &gradient_prev,
			float beta_bias_update, Tensor &bias_update, ActivationType act);

	/*
	 * tensor op
	 */
	void emulateLowPrecision(const Context &context, Tensor &dst, const Tensor &src, DataType dtype, AffineTransform transform);
	void sumOverFirstDim(const Context &context, float alpha, const Tensor &src, float beta, Tensor &dst);
	void multiplyTensors(const Context &context, Tensor &dst, const Tensor &lhs, const Tensor &rhs);
	void addTensors(const Context &context, Tensor &dst, const Tensor &src1, const Tensor &src2);
	// computes dst = alpha1 * src1 + alpha2 * src2 + beta * dst
	void addTensors(const Context &context, float beta, Tensor &dst, float alpha1, const Tensor &src1, float alpha2, const Tensor &src2);

	/*
	 * training
	 */
	float meanSquaredLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask);
	float crossEntropyLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask);
	void meanSquaredGradient(const Context &context, float alpha, const Tensor &output, const Tensor &target, const Tensor &mask, float beta,
			Tensor &gradient);
	void crossEntropyGradient(const Context &context, float alpha, const Tensor &output, const Tensor &target, const Tensor &mask, float beta,
			Tensor &gradient);
	void radamOptimize(const Context &context, float scale, const std::vector<Tensor> &gradients, std::vector<Tensor> &weights,
			std::vector<Tensor> &momentums, std::vector<Tensor> &variances, std::vector<Tensor> &weights_copy, float learning_rate, float beta1,
			float beta2, int step, float weight_decay);
	void lionOptimize(const Context &context, float scale, const std::vector<Tensor> &gradients, std::vector<Tensor> &weights,
			std::vector<Tensor> &momentums, std::vector<Tensor> &weights_copy, float learning_rate, float beta1, float beta2, int step,
			float weight_decay);
	std::vector<int> isNanOrInf(const Context &context, const std::vector<Tensor> &tensors);
	void l2Regularization(const Context &context, std::vector<Tensor> &gradients, const std::vector<Tensor> &params, float scale);

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
			Tensor &workspace, float beta);
	void explicit_gemm_update(const Context &context, const Tensor &input, const Tensor &gradient_next, Tensor &weight_update, Tensor &workspace);

} /* namespace ml */

#endif /* MINML_MATH_HPP_ */
