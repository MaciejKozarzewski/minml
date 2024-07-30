/*
 * attention.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "kernel_table.hpp"

#include <vector>
#include <cmath>
#include <iostream>

#include <CL/opencl.hpp>
#include <clblast.h>
#include <clblast_half.h>

namespace
{
	using namespace ml;

	void calculate_offsets(std::vector<size_t> &q_offsets, std::vector<size_t> &k_offsets, std::vector<size_t> &v_offsets,
			std::vector<size_t> &qk_offsets, std::vector<size_t> &out_offsets, int batch_size, int tokens, int num_heads, int head_dim)
	{
		const int embedding = num_heads * head_dim;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < num_heads; h++)
			{
				const int idx = b * num_heads + h;
				q_offsets[idx] = b * tokens * 3 * embedding + h * head_dim;
				k_offsets[idx] = q_offsets[idx] + embedding;
				v_offsets[idx] = k_offsets[idx] + embedding;
				qk_offsets[idx] = (b * num_heads + h) * tokens * tokens;
				out_offsets[idx] = (b * tokens * num_heads + h) * head_dim;
			}
	}

	void gemm_batched(mlContext_t context, char opA, char opB, mlDataType_t dtype, int M, int N, int K, float alpha, const void *A,
			const std::vector<size_t> &a_offsets, int lda, const void *B, const std::vector<size_t> &b_offsets, int ldb, float beta, void *C,
			const std::vector<size_t> &c_offsets, int ldc, int batch_count)
	{
		assert(context != nullptr);

		clblast::Transpose a_transpose = is_transpose(opA) ? clblast::Transpose::kYes : clblast::Transpose::kNo;
		clblast::Transpose b_transpose = is_transpose(opB) ? clblast::Transpose::kYes : clblast::Transpose::kNo;

		cl::CommandQueue &queue = opencl::Context::getCommandQueue(context);
		const cl::Buffer &a_buffer = opencl::getMemoryObject(A).buffer();
		const cl::Buffer &b_buffer = opencl::getMemoryObject(B).buffer();
		cl::Buffer &c_buffer = opencl::getMemoryObject(C).buffer();
		cl::Event &event = *opencl::Context::getLastEvent(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				std::vector<cl_half> alphas(batch_count, FloatToHalf(alpha));
				std::vector<cl_half> betas(batch_count, FloatToHalf(beta));
				clblast::StatusCode status = clblast::GemmBatched(clblast::Layout::kRowMajor, a_transpose, b_transpose, M, N, K, alphas.data(),
						a_buffer(), a_offsets.data(), lda, b_buffer(), b_offsets.data(), ldb, betas.data(), c_buffer(), c_offsets.data(), ldc,
						batch_count, &queue(), &event());
				assert(status == clblast::StatusCode::kSuccess);
				break;
			}
			case DTYPE_FLOAT32:
			{
				std::vector<float> alphas(batch_count, alpha);
				std::vector<float> betas(batch_count, beta);
				clblast::StatusCode status = clblast::GemmBatched(clblast::Layout::kRowMajor, a_transpose, b_transpose, M, N, K, alphas.data(),
						a_buffer(), a_offsets.data(), lda, b_buffer(), b_offsets.data(), ldb, betas.data(), c_buffer(), c_offsets.data(), ldc,
						batch_count, &queue(), &event());
				assert(status == clblast::StatusCode::kSuccess);
				break;
			}
			default:
				break;
		}
	}
}

namespace ml
{
	int opencl_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, bool training)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int tokens = input_shape.dim[1] * input_shape.dim[2];
		const int num_heads = weights_shape.dim[0];

		int result = batch_size * num_heads * tokens * tokens;
		if (training)
			result = result * 2 + batch_size * num_heads * weights_shape.dim[1] * weights_shape.dim[2];
		return result;
	}
	void opencl_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlDataType_t dtype,
			const void *input, void *output, const void *weights, void *workspace, void *backward_data)
	{
		assert(input_shape.rank == 3);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int tokens = input_shape.dim[1];
		const int embedding = input_shape.dim[2] / 3;
		const int num_heads = weights_shape.dim[0];
		const int head_dim = embedding / num_heads;

		const int num_pointers = batch_size * num_heads;

		std::vector<size_t> q_offsets(num_pointers, 0);
		std::vector<size_t> k_offsets(num_pointers, 0);
		std::vector<size_t> v_offsets(num_pointers, 0);
		std::vector<size_t> qk_offsets(num_pointers, 0);
		std::vector<size_t> out_offsets(num_pointers, 0);
		calculate_offsets(q_offsets, k_offsets, v_offsets, qk_offsets, out_offsets, batch_size, tokens, num_heads, head_dim);

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', dtype, tokens, tokens, head_dim, scale, input, q_offsets, 3 * embedding, input, k_offsets, 3 * embedding,
				0.0f, workspace, qk_offsets, tokens, num_pointers);

		const mlShape_t qk_shape = make_shape( { batch_size * num_heads * tokens, tokens });
		opencl_activation_forward(context, dtype, qk_shape, workspace, workspace, ACTIVATION_SOFTMAX);

		gemm_batched(context, 'n', 'n', dtype, tokens, head_dim, tokens, 1.0f, workspace, qk_offsets, tokens, input, v_offsets, 3 * embedding, 0.0f,
				output, out_offsets, embedding, num_pointers);
	}
	void opencl_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, void *gradient_prev, void *gradient_next, void *weights_update, void *workspace, void *backward_data)
	{
		assert(input_shape.rank == 3);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int tokens = input_shape.dim[1];
		const int embedding = input_shape.dim[2] / 3;
		const int num_heads = weights_shape.dim[0];
		const int head_dim = embedding / num_heads;

		const int num_pointers = batch_size * num_heads;

		std::vector<size_t> q_offsets(num_pointers, 0);
		std::vector<size_t> k_offsets(num_pointers, 0);
		std::vector<size_t> v_offsets(num_pointers, 0);
		std::vector<size_t> qk_offsets(num_pointers, 0);
		std::vector<size_t> out_offsets(num_pointers, 0);
		calculate_offsets(q_offsets, k_offsets, v_offsets, qk_offsets, out_offsets, batch_size, tokens, num_heads, head_dim);
		std::vector<size_t> dqk_offsets = qk_offsets;

		const size_t workspace_offset = batch_size * num_heads * tokens * tokens;
		for (size_t i = 0; i < dqk_offsets.size(); i++)
			dqk_offsets[i] += workspace_offset;

		const float scale = 1.0f / std::sqrt(head_dim);
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, scale, input, q_offsets, 3 * embedding, input, k_offsets,
				3 * embedding, 0.0f, workspace, qk_offsets, tokens, num_pointers);

		const mlShape_t qk_shape = make_shape( { batch_size * num_heads * tokens, tokens });
		opencl_activation_forward(context, DTYPE_FLOAT32, qk_shape, workspace, workspace, ACTIVATION_SOFTMAX);

		// dqk = dy * V^T
		gemm_batched(context, 'n', 't', DTYPE_FLOAT32, tokens, tokens, head_dim, 1.0f, gradient_next, out_offsets, embedding, input, v_offsets,
				3 * embedding, 0.0f, workspace, dqk_offsets, tokens, num_pointers);
		// dV = qk^T * dy
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, workspace, qk_offsets, tokens, gradient_next, out_offsets,
				embedding, 0.0f, gradient_prev, v_offsets, 3 * embedding, num_pointers);

		// run softmax backward pass
		static const ml::opencl::ProgramCache program_cache("activations_backward",
				ml::opencl::kernels::common + ml::opencl::kernels::activations_backward, "");

		cl::Kernel kernel = program_cache.getKernel(context, "attention_softmax_backward_fp32");
		cl::NDRange global = opencl::get_nd_range<65536>(workspace_offset);
		cl::NDRange local = cl::NullRange;

		kernel.setArg(0, opencl::getMemoryObject(workspace).buffer());
		kernel.setArg(1, static_cast<int>(workspace_offset));
		opencl::runKernel(context, kernel, global, local);

		// dQ = dqk * K
		gemm_batched(context, 'n', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, workspace, dqk_offsets, tokens, input, k_offsets,
				3 * embedding, 0.0f, gradient_prev, q_offsets, 3 * embedding, num_pointers);
		// dK = dqk^T * Q
		gemm_batched(context, 't', 'n', DTYPE_FLOAT32, tokens, head_dim, tokens, 1.0f, workspace, dqk_offsets, tokens, input, q_offsets,
				3 * embedding, 0.0f, gradient_prev, k_offsets, 3 * embedding, num_pointers);
	}
} /* namespace ml */

