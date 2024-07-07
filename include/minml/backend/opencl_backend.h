/*
 * opencl_backend.h
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BACKEND_OPENCL_BACKEND_H_
#define MINML_BACKEND_OPENCL_BACKEND_H_

#include <minml/backend/backend_types.h>

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#  define CL_HPP_TARGET_OPENCL_VERSION 200
#endif

namespace ml
{

#ifdef __cplusplus
	extern "C"
	{
#endif

		// implemented in 'opencl_properties.cpp'
		int opencl_get_number_of_devices();
		/*
		 * \brief In MB.
		 */
		int opencl_get_memory(int index);
		bool opencl_supports_type(int index, mlDataType_t dtype);
		const char* opencl_get_device_info(int index);
		const char* opencl_get_device_features(int index);

		// implemented in 'opencl_context.cpp'
		mlContext_t opencl_create_context(int device_index);
		void opencl_synchronize_with_context(mlContext_t context);
		bool opencl_is_context_ready(mlContext_t context);
		void opencl_destroy_context(mlContext_t context);

		// implemented in 'opencl_event.cpp'
		mlEvent_t opencl_create_event(mlContext_t context);
		double opencl_get_time_between_events(mlEvent_t start, mlEvent_t end);
		void opencl_wait_for_event(mlEvent_t event);
		bool opencl_is_event_ready(mlEvent_t event);
		void opencl_destroy_event(mlEvent_t event);

		// implemented in 'opencl_memory.cpp'
		void* opencl_malloc(int device_index, int count);
		void opencl_free(void *ptr);
		void* opencl_create_view(void *src, int offset, int count);
		void opencl_destroy_view(void *ptr);
		void opencl_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count);
		void opencl_memcpy_within_device(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count);
		void opencl_memcpy_from_host(mlContext_t context, void *dst, int dst_offset, const void *src, int count);
		void opencl_memcpy_to_host(mlContext_t context, void *dst, const void *src, int src_offset, int count);

		// implemented in 'conversion.cpp'
		void opencl_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src);
		void opencl_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements);
		void opencl_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output);

		// implemented in 'winograd_non_fused.cpp'
		void opencl_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
				void *matrices, bool invert);
		void opencl_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
				const void *input, void *matrices);
		void opencl_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
				const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act);
		void opencl_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape,
				mlShape_t gradient_shape, const void *gradient, void *matrices);
		void opencl_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
				void *update);

		// implemented in 'implicit_gemm_conv.cpp'
		void opencl_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
				const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act);

		// implemented in 'global_pooling.cpp'
		void opencl_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input);
		void opencl_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
				const void *input, const void *output);
		void opencl_global_broadcasting_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
				const void *bias, mlActivationType_t act);
		void opencl_global_broadcasting_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next, const void *output,
				mlActivationType_t act);

		// implemented in 'gemms.cpp'
		void opencl_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
				const void *B, char opA, char opB, float alpha, float beta);
		void opencl_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
				mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);

		// implemented in 'add_bias_act.cpp'
		void opencl_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
				mlActivationType_t act);

		// batchnorm
		void opencl_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights,
				mlActivationType_t act);
		void opencl_batchnorm_forward(mlContext_t context, mlShape_t shape, const void *input, void *output, void *weights, void *running_stats,
				int running_stat_idx, mlActivationType_t act);
		void opencl_batchnorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev,
				void *gradient_next, const void *weights, void *weights_update, const void *running_stats, int running_stat_idx,
				mlActivationType_t act);
		void opencl_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma, bool use_beta);
		void opencl_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights);

		// layernorm
		void opencl_layernorm_forward(mlContext_t context, mlShape_t shape, mlDataType_t dtype, const void *input, void *output, const void *weights,
				const void *bias, const void *ext);
		void opencl_layernorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev,
				void *gradient_next, const void *weights, void *weights_update, void *bias_update);

		/*
		 * attention
		 */
		int opencl_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, bool training);
		void opencl_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlDataType_t dtype,
				const void *input, void *output, const void *weights, void *workspace);
		void opencl_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
				const void *weights, void *gradient_prev, void *gradient_next, void *weights_update, void *workspace);

		void opencl_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
				mlActivationType_t act);
		void opencl_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
				mlActivationType_t act);

		// implemented in 'training.cpp'
		void opencl_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src);
		void opencl_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2);
		void opencl_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta);
		float opencl_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		void opencl_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight);
		float opencl_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		void opencl_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target,
				float weight);
		void opencl_adam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
				float learning_rate, float beta1, float beta2);
		void opencl_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset);

#ifdef __cplusplus
	}
#endif
} /* namespace ml */

#endif /* MINML_BACKEND_OPENCL_BACKEND_H_ */
