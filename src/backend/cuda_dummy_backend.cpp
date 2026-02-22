/*
 * cuda_dummy_backend.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */
#ifdef USE_CUDA
#  ifdef USE_OPENCL
#    error "CUDA and OPENCL cannot be used at the same time"
#  endif
#else
#include <minml/backend/cuda_backend.h>
#include <minml/core/ml_exceptions.hpp>

namespace ml
{
	int cuda_get_number_of_devices()
	{
		return 0;
	}
	void cuda_enable_tf32(mlContext_t context, bool b)
	{
	}
	/*
	 * \brief In MB.
	 */
	int cuda_get_memory(int index)
	{
		throw NotImplemented(METHOD_NAME);
	}
	bool cuda_supports_type(int index, mlDataType_t dtype)
	{
		throw NotImplemented(METHOD_NAME);
	}
	const char* cuda_get_device_info(int index)
	{
		throw NotImplemented(METHOD_NAME);
	}
	const char* cuda_get_device_features(int index)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'cuda_context.cpp'
	mlContext_t cuda_create_context(int device_index)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_synchronize_with_context(mlContext_t context)
	{
		throw NotImplemented(METHOD_NAME);
	}
	bool cuda_is_context_ready(mlContext_t context)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_destroy_context(mlContext_t context)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'cuda_event.cpp'
	mlEvent_t cuda_create_event(mlContext_t context)
	{
		throw NotImplemented(METHOD_NAME);
	}
	double cuda_get_time_between_events(mlEvent_t start, mlEvent_t end)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_wait_for_event(mlEvent_t event)
	{
		throw NotImplemented(METHOD_NAME);
	}
	bool cuda_is_event_ready(mlEvent_t event)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_destroy_event(mlEvent_t event)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'cuda_memory.cu'
	void* cuda_malloc(int device_index, size_t count)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_page_lock(void *ptr, size_t count)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_page_unlock(void *ptr)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_free(void *ptr)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void* cuda_create_view(void *src, size_t offset, size_t count)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_destroy_view(void *ptr)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_memset(mlContext_t context, void *dst, size_t dst_offset, size_t dst_count, const void *src, size_t src_count)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_memcpy_within_device(mlContext_t context, void *dst, size_t dst_offset, const void *src, size_t src_offset, size_t count)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_memcpy_from_host(mlContext_t context, void *dst, size_t dst_offset, const void *src, size_t count)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_memcpy_to_host(mlContext_t context, void *dst, const void *src, size_t src_offset, size_t count)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'conversion.cu'
	void cuda_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_space_to_depth(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, const void *input, mlShape_t output_shape, void *output)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_depth_to_space(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, const void *input, mlShape_t output_shape, void *output)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'winograd_non_fused.cu'
	void cuda_winograd_weight_transform(mlContext_t context, int tile_size, const mlTensor_t w, mlTensor_t matrices, bool invert)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_winograd_input_transform(mlContext_t context, int tile_size, const mlTensor_t x, mlTensor_t matrices)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_winograd_output_transform(mlContext_t context, int tile_size, const mlTensor_t matrices, const mlTensor_t bias, const mlTensor_t ext,
			mlTensor_t y, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_winograd_gradient_transform(mlContext_t context, int tile_size, const mlTensor_t dy, mlTensor_t matrices)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_winograd_update_transform(mlContext_t context, int tile_size, const mlTensor_t matrices, mlTensor_t dw)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'implicit_gemm_conv.cu'
	void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cudnn_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, const void *bias, void *output)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cudnn_depthwise_conv_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *gradient_next,
			const void *weights, void *gradient_prev)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cudnn_depthwise_conv_update(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *gradient_next, void *weights_update)
	{
		throw NotImplemented(METHOD_NAME);
	}

	void cuda_depthwise_conv_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_depthwise_conv_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t w, float beta, mlTensor_t dx)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_depthwise_conv_update(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t dy, float beta, mlTensor_t dw)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'winograd_fused.cu'
	void cuda_convolution_fused_winograd_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}

	void cuda_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y, int size)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx, int size)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'global_pooling.cu'
	void cuda_global_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_global_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_channel_scaling_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t scales, float beta, mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_channel_scaling_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t x, const mlTensor_t scales,
			float beta_dx, mlTensor_t dx, float beta_scales, mlTensor_t dscales)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_channel_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_channel_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_spatial_scaling_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t scales, float beta, mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_spatial_scaling_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t x, const mlTensor_t scales,
			float beta_dx, mlTensor_t dx, float beta_scales, mlTensor_t dscales)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'gemms.cpp'
	void cuda_gemm_v2(mlContext_t context, char opA, char opB, float alpha, const mlTensor_t A, const mlTensor_t B, float beta, mlTensor_t C)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_gemm_batched_v2(mlContext_t context, char opA, char opB, float alpha, const mlTensor_t A, const mlTensor_t B, float beta, mlTensor_t C)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		throw NotImplemented(METHOD_NAME);
	}

	void cuda_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, const void *bias, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// implemented in 'add_bias_act.cu'
	void cuda_add_bias_act(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t b, float beta, mlTensor_t y, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// batchnorm
	void cuda_batchnorm_inference(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b,
			const mlTensor_t avg_var, float beta, mlTensor_t y, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_batchnorm_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta,
			mlTensor_t y, mlTensor_t running_stats, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_batchnorm_backward(mlContext_t context, float alpha, const mlTensor_t x, mlTensor_t dy, const mlTensor_t w, const mlTensor_t b,
			const mlTensor_t running_stats, float beta_dx, mlTensor_t dx, float beta_dw, mlTensor_t dw, mlTensor_t db, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_batchnorm_update(mlContext_t context, const mlTensor_t running_stat, mlTensor_t avg_var)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_fold_batchnorm(mlContext_t context, mlTensor_t layer_weights, mlTensor_t layer_bias, const mlTensor_t bn_weights,
			const mlTensor_t bn_bias, const mlTensor_t bn_avg_var)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// layernorm
	void cuda_layernorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights,
			const void *bias, const void *ext)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_layernorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next,
			const void *weights, void *weights_update, void *bias_update)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// layernorm
	void cuda_rmsnorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_rmsnorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next, const void *weights,
			void *weights_update)
	{
		throw NotImplemented(METHOD_NAME);
	}

	// attention
	int cuda_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, int num_heads, bool training)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlShape_t bias_shape,
			mlDataType_t dtype, const void *input, void *output, const void *weights, const void *bias, const void *mask, void *workspace,
			void *backward_data, int num_heads, bool symmetric)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlShape_t bias_shape,
			const void *input, const void *weights, const void *bias, const void *mask, void *gradient_prev, void *gradient_next,
			void *weights_update, void *bias_update, void *mask_update, void *workspace, void *backward_data, int num_heads, bool symmetric)
	{
		throw NotImplemented(METHOD_NAME);
	}

	void cuda_window_partitioning(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, const void *input,
			void *output, mlShape_t offset)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_window_merging(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, const void *input, void *output,
			mlShape_t offset)
	{
		throw NotImplemented(METHOD_NAME);
	}

	void cuda_activation_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_activation_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t y, float beta, mlTensor_t dx,
			mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_softmax_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_softmax_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t y, float beta, mlTensor_t dx)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_fused_act_bias_copy_backward(mlContext_t context, mlTensor_t dy, const mlTensor_t y, float beta_dx, mlTensor_t dx, float beta_dw,
			mlTensor_t dw, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}

	/*
	 * mixture of experts
	 */
	void cuda_select_top_k(mlContext_t context, const mlTensor_t x, mlTensor_t indices, mlTensor_t values)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_gather_tokens_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices, float beta, mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_gather_tokens_backward(mlContext_t context, const mlTensor_t dy, const mlTensor_t indices, float beta, mlTensor_t dx)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_scatter_tokens_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices, mlTensor_t scales, float beta,
			mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_scatter_tokens_backward(mlContext_t context, const mlTensor_t x, const mlTensor_t indices, const mlTensor_t scales, const mlTensor_t dy,
			float beta1, mlTensor_t dx, float beta2, mlTensor_t dscales)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_moe_forward(mlContext_t context, const mlTensor_t x, const mlTensor_t w, const mlTensor_t b, float beta, mlTensor_t y,
			mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_moe_backward(mlContext_t context, const mlTensor_t x, const mlTensor_t y, const mlTensor_t w, mlTensor_t dy, float beta_dx,
			mlTensor_t dx, float beta_dw, mlTensor_t dw, float beta_db, mlTensor_t db, mlActivationType_t act)
	{
		throw NotImplemented(METHOD_NAME);
	}

	/*
	 * tensor op
	 */
	void cuda_emulate_low_precision(mlContext_t context, mlShape_t shape, mlDataType_t dtype, void *dst, const void *src, mlQuantizationData_t qd)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_scale_tensor(mlContext_t context, float alpha, mlTensor_t x)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_add_tensors(mlContext_t context, float alpha1, const mlTensor_t x1, float alpha2, const mlTensor_t x2, float alpha3,
			const mlTensor_t x3, float beta, mlTensor_t y)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_sum_over_first_dim(mlContext_t context, float alpha, const mlTensor_t src, float beta, mlTensor_t dst)
	{
		throw NotImplemented(METHOD_NAME);
	}

	/*
	 * training
	 */
	float cuda_mean_squared_loss(mlContext_t context, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask)
	{
		throw NotImplemented(METHOD_NAME);
	}
	float cuda_cross_entropy_loss(mlContext_t context, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_mean_squared_gradient(mlContext_t context, float alpha, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask,
			float beta, mlTensor_t gradient)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_cross_entropy_gradient(mlContext_t context, float alpha, const mlTensor_t output, const mlTensor_t target, const mlTensor_t mask,
			float beta, mlTensor_t gradient)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_fused_radam_optimize(mlContext_t context, float scale, const mlTensor_t *gradients, mlTensor_t *weights, mlTensor_t *momentums,
			mlTensor_t *variances, mlTensor_t *weights_copy, float learning_rate, float beta1, float beta2, int step, int num_tensors,
			float weight_decay)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_fused_lion_optimize(mlContext_t context, float scale, const mlTensor_t *gradients, mlTensor_t *weights, mlTensor_t *momentums,
			mlTensor_t *weights_copy, float learning_rate, float beta1, float beta2, int step, int num_tensors, float weight_decay)
	{
		throw NotImplemented(METHOD_NAME);
	}
	int cuda_is_nan_or_inf(mlContext_t context, const mlTensor_t tensor)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_l2_regularization(mlContext_t context, mlTensor_t gradient, const mlTensor_t param, float coefficient, float offset)
	{
		throw NotImplemented(METHOD_NAME);
	}

	void cuda_fused_radam_optimize(mlContext_t context, float scale, const mlTensor_t *gradients, mlTensor_t *weights, mlTensor_t *momentums,
			mlTensor_t *variances, float learning_rate, float beta1, float beta2, int step, int num_tensors)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_fused_is_nan_or_inf(mlContext_t context, const mlTensor_t *tensors, int *result, int num_tensors)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_fused_l2_regularization(mlContext_t context, mlTensor_t *gradients, const mlTensor_t *params, float scale, int num_tensors)
	{
		throw NotImplemented(METHOD_NAME);
	}

	/*
	 * quantization
	 */
	void cuda_dequantize(mlContext_t context, mlDataType_t dtype, const void *input, void *output, int elements, float scale, float shift)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_quantized_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, const void *scales, const void *bias, void *output, mlQuantizationData_t output_qd,
			int padding_value)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_quantized_scale_shift_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, mlQuantizationData_t output_qd,
			const void *input, const void *scales, const void *bias, mlActivationType_t act, const void *ext, mlQuantizationData_t ext_qd)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_im2row(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, void *output, const void *input, int kernel_size, bool invert,
			const void *padding_value)
	{
		throw NotImplemented(METHOD_NAME);
	}
	void cuda_transpose(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t output_shape, void *output, const void *input,
			const int *ordering)
	{
		throw NotImplemented(METHOD_NAME);
	}

} /* namespace ml */

#endif
