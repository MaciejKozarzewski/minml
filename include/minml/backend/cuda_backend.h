/*
 * cuda_backend.h
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BACKEND_CUDA_BACKEND_H_
#define MINML_BACKEND_CUDA_BACKEND_H_

#include <minml/backend/backend_types.h>

namespace ml
{

#ifdef __cplusplus
	extern "C"
	{
#endif

		// implemented in 'cuda_properties.cpp'
		DLL_PUBLIC int cuda_get_number_of_devices();
		/*
		 * \brief In MB.
		 */
		DLL_PUBLIC int cuda_get_memory(int index);
		DLL_PUBLIC bool cuda_supports_type(int index, mlDataType_t dtype);
		DLL_PUBLIC const char* cuda_get_device_info(int index);
		DLL_PUBLIC const char* cuda_get_device_features(int index);

		// implemented in 'cuda_context.cpp'
		DLL_PUBLIC mlContext_t cuda_create_context(int device_index);
		DLL_PUBLIC void cuda_synchronize_with_context(mlContext_t context);
		DLL_PUBLIC bool cuda_is_context_ready(mlContext_t context);
		DLL_PUBLIC void cuda_destroy_context(mlContext_t context);

		// implemented in 'cuda_event.cpp'
		DLL_PUBLIC mlEvent_t cuda_create_event(mlContext_t context);
		DLL_PUBLIC double cuda_get_time_between_events(mlEvent_t start, mlEvent_t end);
		DLL_PUBLIC void cuda_wait_for_event(mlEvent_t event);
		DLL_PUBLIC bool cuda_is_event_ready(mlEvent_t event);
		DLL_PUBLIC void cuda_destroy_event(mlEvent_t event);

		// implemented in 'cuda_memory.cu'
		DLL_PUBLIC void* cuda_malloc(int device_index, int count);
		DLL_PUBLIC void cuda_page_lock(void *ptr, int count);
		DLL_PUBLIC void cuda_page_unlock(void *ptr);
		DLL_PUBLIC void cuda_free(void *ptr);
		DLL_PUBLIC void* cuda_create_view(void *src, int offset, int count);
		DLL_PUBLIC void cuda_destroy_view(void *ptr);
		DLL_PUBLIC void cuda_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count);
		DLL_PUBLIC void cuda_memcpy_within_device(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count);
		DLL_PUBLIC void cuda_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count);
		DLL_PUBLIC void cuda_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count);

		// implemented in 'conversion.cu'
		DLL_PUBLIC void cuda_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src);
		DLL_PUBLIC void cuda_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype,
				int elements);
		DLL_PUBLIC void cuda_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output);

		// implemented in 'winograd_non_fused.cu'
		DLL_PUBLIC void cuda_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
				const void *weights, void *matrices, bool invert);
		DLL_PUBLIC void cuda_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
				mlShape_t input_shape, const void *input, void *matrices);
		DLL_PUBLIC void cuda_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
				mlShape_t output_shape, const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act);
		DLL_PUBLIC void cuda_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
				mlShape_t gradient_shape, const void *gradient, void *matrices);
		DLL_PUBLIC void cuda_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
				const void *matrices, void *update);

		// implemented in 'implicit_gemm_conv.cu'
		DLL_PUBLIC void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape,
				mlShape_t weights_shape, const void *input, const void *weights, void *output, const void *bias, const void *add,
				mlActivationType_t act);

		// implemented in 'global_pooling.cu'
		DLL_PUBLIC void cuda_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output,
				const void *input);
		DLL_PUBLIC void cuda_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
				const void *input, const void *output);
		DLL_PUBLIC void cuda_global_broadcasting_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
				const void *bias, mlActivationType_t act);
		DLL_PUBLIC void cuda_global_broadcasting_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next,
				const void *output, mlActivationType_t act);

		// implemented in 'gemms.cpp'
		DLL_PUBLIC void cuda_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
				mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);
		DLL_PUBLIC void cuda_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
				mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);

		/*
		 * Computes D = act(alpha * op_A(A) * op_B(B) + beta * C + bias)
		 */
		DLL_PUBLIC void cuda_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A,
				const void *A, char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, const void *bias,
				mlActivationType_t act);

		// implemented in 'add_bias_act.cu'
		DLL_PUBLIC void cuda_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
				mlActivationType_t act);

		// batchnorm
		DLL_PUBLIC void cuda_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights,
				mlActivationType_t act);
		DLL_PUBLIC void cuda_batchnorm_forward(mlContext_t context, mlShape_t shape, const void *input, void *output, void *weights,
				void *running_stats, int running_stat_idx, mlActivationType_t act);
		DLL_PUBLIC void cuda_batchnorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev,
				void *gradient_next, const void *weights, void *weights_update, const void *running_stats, int running_stat_idx,
				mlActivationType_t act);
		DLL_PUBLIC void cuda_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma,
				bool use_beta);
		DLL_PUBLIC void cuda_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias,
				const void *batchnorm_weights);

		// layernorm
		DLL_PUBLIC void cuda_layernorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output,
				const void *weights, const void *bias, const void *ext);
		DLL_PUBLIC void cuda_layernorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next,
				const void *weights, void *weights_update, void *bias_update);

		/*
		 * RMSnorm
		 */
		DLL_PUBLIC void cuda_rmsnorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output,
				const void *weights);
		DLL_PUBLIC void cuda_rmsnorm_backward(mlContext_t context, mlShape_t shape, const void *input, void *gradient_prev, void *gradient_next,
				const void *weights, void *weights_update);

		/*
		 * attention
		 */
		DLL_PUBLIC int cuda_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, bool training);
		DLL_PUBLIC void cuda_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlDataType_t dtype,
				const void *input, void *output, const void *weights, void *workspace, void *backward_data);
		DLL_PUBLIC void cuda_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
				const void *weights, void *gradient_prev, void *gradient_next, void *weights_update, void *workspace, void *backward_data);

		// activations
		DLL_PUBLIC void cuda_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
				mlActivationType_t act);
		DLL_PUBLIC void cuda_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
				const void *output, mlActivationType_t act);

		// implemented in 'training.cu'
		DLL_PUBLIC void cuda_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src);
		DLL_PUBLIC void cuda_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2);
		DLL_PUBLIC void cuda_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2);
		DLL_PUBLIC void cuda_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta);
		DLL_PUBLIC float cuda_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		DLL_PUBLIC void cuda_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target,
				float weight);
		DLL_PUBLIC float cuda_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		DLL_PUBLIC void cuda_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target,
				float weight);
		DLL_PUBLIC void cuda_radam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
				float learning_rate, float beta1, float beta2, int step);
		DLL_PUBLIC void cuda_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient,
				float offset);

#ifdef __cplusplus
	}
#endif
} /* namespace ml */

#endif /* MINML_BACKEND_CUDA_BACKEND_H_ */
