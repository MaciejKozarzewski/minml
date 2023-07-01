/*
 * cpu_backend.h
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BACKEND_CPU_BACKEND_H_
#define MINML_BACKEND_CPU_BACKEND_H_

#include <minml/backend/backend_types.h>

namespace ml
{

#ifdef __cplusplus
	extern "C"
	{
#endif

		void cpu_set_number_of_threads(int number);
		int cpu_get_number_of_cores();
		/*
		 * \brief In MB.
		 */
		int cpu_get_memory();
		int cpu_get_simd_level();
		bool cpu_supports_type(mlDataType_t dtype);
		const char* cpu_get_device_info();
		void cpu_print_device_features();

		mlContext_t cpu_create_context();
		void cpu_synchronize_with_context(mlContext_t context);
		bool cpu_is_context_ready(mlContext_t context);
		void cpu_destroy_context(mlContext_t context);

		void* cpu_malloc(int count);
		void cpu_free(void *ptr);
		void* cpu_view(void *src, int offset, int count);

		void cpu_memset(mlContext_t context, void *dst, int dst_offset, int dst_count, const void *src, int src_count);
		void cpu_memcpy(mlContext_t context, void *dst, int dst_offset, const void *src, int src_offset, int count);

		void cpu_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src);
		void cpu_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements);
		void cpu_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output);

		void cpu_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
				void *matrices, bool invert);
		void cpu_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
				const void *input, void *matrices);
		void cpu_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
				const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act);
		void cpu_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
				const void *gradient, void *matrices);
		void cpu_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
				void *update);

		void cpu_im2row(mlContext_t context, mlDataType_t dtype, mlShape_t weights_shape, mlShape_t input_shape, const void *input, void *matrix);

		void cpu_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
				const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act);

		void cpu_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
				const void *B, char opA, char opB, float alpha, float beta);
		void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
				mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);

		/*
		 * Computes D = act(alpha * op_A(A) * op_B(B) + beta * C)
		 */
		void cpu_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
				char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, mlActivationType_t act);

		void cpu_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act);

		void cpu_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights,
				mlActivationType_t act);
		void cpu_batchnorm_forward(mlContext_t context, mlShape_t shape, const void *input, void *output, void *weights, void *running_stats,
				int running_stat_idx, mlActivationType_t act);
		void cpu_batchnorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev,
				void *gradient_next, const void *weights, void *weights_update, const void *running_stats, int running_stat_idx,
				mlActivationType_t act);
		void cpu_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma, bool use_beta);
		void cpu_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights);

		void cpu_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
				mlActivationType_t act);
		void cpu_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
				mlActivationType_t act);

		// implemented in 'global_pooling.cu'
		void cpu_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output,
				const void *weights);
		void cpu_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
				const void *input, const void *weights);

		// used for training
		void cpu_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src);
		void cpu_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2);
		void cpu_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta);
		float cpu_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		void cpu_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight);
		float cpu_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target);
		void cpu_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight);
		void cpu_adam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
				float learning_rate, float beta1, float beta2);
		void cpu_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset);

#ifdef __cplusplus
	}
#endif
} /* namespace ml */

#endif /* MINML_BACKEND_CPU_BACKEND_H_ */
