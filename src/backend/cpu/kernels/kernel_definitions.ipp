/*
 * kernel_definitions.ipp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

using ml::mlContext_t;
using ml::mlShape_t;
using ml::mlDataType_t;
using ml::mlActivationType_t;

void cpu_kernel_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src);
void cpu_kernel_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements);

void cpu_kernel_winograd_weight_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, const void *weights, void *matrices,
		bool invert, bool low_precision);
void cpu_kernel_winograd_input_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape, const void *input,
		void *matrices);
void cpu_kernel_winograd_output_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
		const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act);
void cpu_kernel_winograd_gradient_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
		const void *gradient, void *matrices);
void cpu_kernel_winograd_update_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices, void *update);

void cpu_kernel_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
		const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act);

void cpu_kernel_convolution_fused_winograd_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
		const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act);

// implemented in 'global_pooling.cpp'
void cpu_kernel_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output,
		void *max_indices);
void cpu_kernel_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
		const void *max_indices);

void cpu_kernel_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
		const void *B, char opA, char opB, float alpha, float beta);
void cpu_kernel_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
		mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta);

void cpu_kernel_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act);

void cpu_kernel_activation_forward_in_place(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, mlActivationType_t act);
void cpu_kernel_activation_backward_in_place(mlContext_t context, mlShape_t shape, void *gradient, const void *output, mlActivationType_t act);