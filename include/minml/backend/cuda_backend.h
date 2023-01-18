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
		int cuda_get_memory(int index);
		DLL_PUBLIC bool cuda_supports_type(int index, mlDataType_t dtype);
		DLL_PUBLIC const char* cuda_get_device_info(int index);

		// implemented in 'cuda_context.cpp'
		DLL_PUBLIC mlContext_t cuda_create_context(int device_index);
		DLL_PUBLIC void cuda_synchronize_with_context(mlContext_t context);
		DLL_PUBLIC bool cuda_is_context_ready(mlContext_t context);
		DLL_PUBLIC void cuda_destroy_context(mlContext_t context);

		// implemented in 'cuda_memory.cu'
		DLL_PUBLIC void* cuda_malloc(int device_index, int count);
		DLL_PUBLIC void cuda_page_lock(void *ptr, int count);
		DLL_PUBLIC void cuda_page_unlock(void *ptr);
		DLL_PUBLIC void cuda_free(void *ptr);
		DLL_PUBLIC void* cuda_view(void *src, int offset, int count);
		DLL_PUBLIC void cuda_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count);
		DLL_PUBLIC void cuda_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count);
		DLL_PUBLIC void cuda_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count);

		// implemented in 'conversion.cu'
		DLL_PUBLIC void cuda_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src);
		DLL_PUBLIC void cuda_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype,
				int elements);

		// implemented in 'winograd_non_fused.cu'
		DLL_PUBLIC void cuda_winograd_weight_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
				void *matrices, bool invert, bool low_precision);
		DLL_PUBLIC void cuda_winograd_input_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
				const void *input, void *matrices);
		DLL_PUBLIC void cuda_winograd_output_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
				const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act);
		DLL_PUBLIC void cuda_winograd_gradient_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
				const void *gradient, void *matrices);
		DLL_PUBLIC void cuda_winograd_update_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
				void *update);

		// implemented in 'implicit_gemm_conv.cu'
		DLL_PUBLIC void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape,
				mlShape_t weights_shape, const void *input, const void *weights, void *output, const void *bias, const void *add,
				mlActivationType_t act);

		// implemented in 'winograd_fused.cu'
		DLL_PUBLIC void cuda_convolution_fused_winograd_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape,
				mlShape_t weights_shape, const void *input, const void *weights, void *output, const void *bias, const void *add,
				mlActivationType_t act);

		// implemented in 'global_pooling.cu'
		DLL_PUBLIC void cuda_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input,
				void *output, void *max_indices);
		DLL_PUBLIC void cuda_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
				const void *max_indices);

		// implemented in 'gemms.cpp'
		DLL_PUBLIC void cuda_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
				mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);
		DLL_PUBLIC void cuda_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
				mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);

		// implemented in 'add_bias_act.cu'
		DLL_PUBLIC void cuda_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias,
				mlActivationType_t act);

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

		DLL_PUBLIC void cuda_activation_forward_in_place(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input,
				mlActivationType_t act);
		DLL_PUBLIC void cuda_activation_backward_in_place(mlContext_t context, mlShape_t shape, void *gradient, const void *output,
				mlActivationType_t act);

		// implemented in 'training.cu'
		DLL_PUBLIC void cuda_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src);
		DLL_PUBLIC void cuda_add_tensors(mlContext_t context, mlShape_t shape, void *dst, const void *src1, const void *src2);
		DLL_PUBLIC void cuda_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta);
		DLL_PUBLIC float cuda_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		DLL_PUBLIC void cuda_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target);
		DLL_PUBLIC void cuda_adam_optimize(mlContext_t context, mlShape_t shape, void *weight, void *update, void *momentum, void *variance,
				float learning_rate, float beta1, float beta2);
		DLL_PUBLIC void cuda_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient,
				float offset);

#ifdef __cplusplus
	}
#endif
} /* namespace ml */

#endif /* MINML_BACKEND_CUDA_BACKEND_H_ */
