/*
 * attention.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "gemm/Matrix.hpp"
#include "gemm/gemm_runtime.hpp"

namespace
{
	using namespace ml;

	class MatrixSlicer
	{
			const uint8_t *q_ptr;
			const uint8_t *k_ptr;
			const uint8_t *v_ptr;
			uint8_t *qk_ptr;
			uint8_t *output_ptr;

			int batch_strides[3]; // input, QK, output
			int head_strides[3]; // input, QK, output

//			int tokens;
//			int embedding_dim;
//			int num_heads;
//			int head_dim;
//			int dtype_size;
//			int batch_qkv_stride;
//			int batch_qk_stride;
//			int batch_output_stride;
		public:
			MatrixSlicer(const mlShape_t &shape, mlDataType_t dtype, const void *input, void *workspace, void *output, int num_heads)
//					tokens(shape.dim[1]),
//					embedding_dim(shape.dim[2] / 3),
//					dtype_size(size_of(dtype)),
//					q_ptr(reinterpret_cast<const uint8_t*>(input)),
//					k_ptr(q_ptr + embedding_dim * dtype_size),
//					v_ptr(k_ptr + embedding_dim * dtype_size),
//					qk_ptr(reinterpret_cast<uint8_t*>(workspace)),
//					output_ptr(reinterpret_cast<uint8_t*>(output)),
//					num_heads(num_heads),
//					head_dim(embedding_dim / num_heads),
//					batch_qkv_stride(shape.dim[1] * shape.dim[2] * dtype_size)
			{
				assert(shape.rank == 3);
				assert(shape.dim[2] % 3 == 0);
				const int batch_size = shape.dim[0];
				const int tokens = shape.dim[1];
				const int embedding_dim = shape.dim[2] / 3;
				const int dtype_size = size_of(dtype);

				assert(num_heads > 0);
				assert(embedding_dim % num_heads == 0);
				const int head_dim = embedding_dim / num_heads;

				q_ptr = reinterpret_cast<const uint8_t*>(input);
				k_ptr = q_ptr + embedding_dim * dtype_size;
				v_ptr = k_ptr + embedding_dim * dtype_size;

				qk_ptr = reinterpret_cast<uint8_t*>(workspace);
				output_ptr = reinterpret_cast<uint8_t*>(output);

				batch_strides[0] = tokens * embedding_dim * 3 * dtype_size;
				batch_strides[1] = tokens * tokens * dtype_size;
				batch_strides[2] = tokens * embedding_dim * dtype_size;

				head_strides[0] = embedding_dim * 3 * dtype_size;

			}
			Matrix get_Q_head(int batch_idx, int head_idx) const noexcept
			{

			}

	};
}

namespace ml
{
	int cpu_multi_head_attention_get_workspace_size(mlShape_t input_shape, mlShape_t weights_shape, bool training)
	{
		return 0;
	}

	void cpu_multi_head_attention_forward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, mlDataType_t dtype,
			const void *input, void *output, const void *weights, void *workspace, void *backward_data)
	{
//		assert(shape.rank == 3);
//		const int batch_size = shape.dim[0];
//		const int tokens = shape.dim[1];
//		assert(shape.dim[2] % 3 == 0);
//		const int embedding_dim = shape.dim[2] / 3;
//
//		assert(num_heads > 0);
//		assert(embedding_dim % num_heads == 0);
//		const int head_dim = embedding_dim / num_heads;
//
//		const mlShape_t head_shape = make_shape( { tokens, head_dim });
//		GemmRuntime rt = get_runtime(context, dtype, 'n', head_shape, 't', head_shape);
//
//		const int batch_stride = shape.dim[1] * shape.dim[2];
//		for (int b = 0; b < batch_size; b++)
//		{
//			const void *q_ptr = reinterpret_cast<const uint8_t*>(input) + 0;
//		}
//		for (int i = 0; i < shape_A.dim[0]; i++)
//		{
//			rt.setMatrixA(getPointer<uint8_t>(A) + i * stride_A, shape_A, dtype, opA);
//			rt.setMatrixB(getPointer<uint8_t>(B) + i * stride_B, shape_B, dtype, opB);
//			rt.setMatrixC(getPointer<uint8_t>(C) + i * stride_C, shape_C, dtype);
//			rt.setMatrixD(getPointer<uint8_t>(C) + i * stride_D, shape_C, dtype);
//			if (i == 0)
//			{
//				rt.setScalingFactors(alpha, beta);
//				rt.setup(context);
//			}
//			rt.run();
//		}

	}
	void cpu_multi_head_attention_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, void *gradient_prev, void *gradient_next, void *weights_update, void *workspace, void *backward_data)
	{
	}
} /* namespace ml */
