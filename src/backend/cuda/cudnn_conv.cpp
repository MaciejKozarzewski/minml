/*
 * cudnn_conv.cpp
 *
 *  Created on: Apr 7, 2023
 *      Author: Maciej Kozarzewski
 */

#ifdef USE_CUDNN

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"

#include "cudnn_ops_infer.h"
#include "cudnn_cnn_infer.h"

#include <cassert>

namespace
{
	using namespace ml;
	cudnnDataType_t convert(mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case DTYPE_UNKNOWN:
				return CUDNN_DATA_FLOAT;
			case DTYPE_BFLOAT16:
				return CUDNN_DATA_BFLOAT16;
			case DTYPE_FLOAT16:
				return CUDNN_DATA_HALF;
			case DTYPE_FLOAT32:
				return CUDNN_DATA_FLOAT;
			case DTYPE_INT32:
				return CUDNN_DATA_INT32;
		}
	}

	cudnnActivationMode_t convert(mlActivationType_t act) noexcept
	{
		switch (act)
		{
			default:
			case ACTIVATION_LINEAR:
				return CUDNN_ACTIVATION_IDENTITY;
			case ACTIVATION_SIGMOID:
				return CUDNN_ACTIVATION_SIGMOID;
			case ACTIVATION_TANH:
				return CUDNN_ACTIVATION_TANH;
			case ACTIVATION_RELU:
				return CUDNN_ACTIVATION_RELU;
		}
	}

	class TensorDescriptor
	{
			cudnnTensorDescriptor_t m_desc = 0;
		public:
			TensorDescriptor() noexcept = default;
			TensorDescriptor(mlShape_t shape, mlDataType_t dtype)
			{
				assert(shape.rank == 4);
				cudnnStatus_t status = cudnnCreateTensorDescriptor(&m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
				status = cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NHWC, convert(dtype), shape.dim[0], shape.dim[3], shape.dim[1],
						shape.dim[2]);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			TensorDescriptor(const TensorDescriptor &other) = delete;
			TensorDescriptor(TensorDescriptor &&other) noexcept = delete;
			TensorDescriptor& operator=(const TensorDescriptor &other) = delete;
			TensorDescriptor& operator=(TensorDescriptor &&other) noexcept = delete;
			~TensorDescriptor() noexcept
			{
				cudnnStatus_t status = cudnnDestroyTensorDescriptor(m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			operator cudnnTensorDescriptor_t() const noexcept
			{
				return m_desc;
			}
	};

	class FilterDescriptor
	{
			cudnnFilterDescriptor_t m_desc = 0;
		public:
			FilterDescriptor() noexcept = default;
			FilterDescriptor(mlShape_t shape, mlDataType_t dtype)
			{
				assert(shape.rank == 4);
				cudnnStatus_t status = cudnnCreateFilterDescriptor(&m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
				status = cudnnSetFilter4dDescriptor(m_desc, convert(dtype), CUDNN_TENSOR_NHWC, shape.dim[0], shape.dim[3], shape.dim[1],
						shape.dim[2]);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			FilterDescriptor(const FilterDescriptor &other) = delete;
			FilterDescriptor(FilterDescriptor &&other) noexcept = delete;
			FilterDescriptor& operator=(const FilterDescriptor &other) = delete;
			FilterDescriptor& operator=(FilterDescriptor &&other) noexcept = delete;
			~FilterDescriptor() noexcept
			{
				cudnnStatus_t status = cudnnDestroyFilterDescriptor(m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			operator cudnnFilterDescriptor_t() const noexcept
			{
				return m_desc;
			}
	};

	class ActivationDescriptor
	{
			cudnnActivationDescriptor_t m_desc = 0;
		public:
			ActivationDescriptor() noexcept = default;
			ActivationDescriptor(mlActivationType_t act)
			{
				cudnnStatus_t status = cudnnCreateActivationDescriptor(&m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
				status = cudnnSetActivationDescriptor(m_desc, convert(act), CUDNN_PROPAGATE_NAN, 0.0f);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			ActivationDescriptor(const ActivationDescriptor &other) = delete;
			ActivationDescriptor(ActivationDescriptor &&other) noexcept = delete;
			ActivationDescriptor& operator=(const ActivationDescriptor &other) = delete;
			ActivationDescriptor& operator=(ActivationDescriptor &&other) noexcept = delete;
			~ActivationDescriptor() noexcept
			{
				cudnnStatus_t status = cudnnDestroyActivationDescriptor(m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			operator cudnnActivationDescriptor_t() const noexcept
			{
				return m_desc;
			}
	};

	class ConvolutionDescriptor
	{
			cudnnConvolutionDescriptor_t m_desc = 0;
			cudnnConvolutionFwdAlgo_t m_algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
			size_t m_workspace_size = 0;
		public:
			ConvolutionDescriptor() noexcept = default;
			ConvolutionDescriptor(cudnnHandle_t handle, const TensorDescriptor &xDesc, const FilterDescriptor &wDesc, const TensorDescriptor &yDesc,
					mlDataType_t dtype, int kernel_size)
			{
				cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
				status = cudnnSetConvolutionMathType(m_desc, CUDNN_TENSOR_OP_MATH);
				assert(status == CUDNN_STATUS_SUCCESS);
				const int padding = kernel_size / 2;
				status = cudnnSetConvolution2dDescriptor(m_desc, padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, convert(dtype));
				assert(status == CUDNN_STATUS_SUCCESS);

				cudnnConvolutionFwdAlgoPerf_t perf;
				int actual_count = 0;
				status = cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, m_desc, yDesc, 1, &actual_count, &perf);
				assert(status == CUDNN_STATUS_SUCCESS);
				assert(actual_count > 0);
				m_algo = perf.algo;
				m_workspace_size = perf.memory;
			}
			ConvolutionDescriptor(const ConvolutionDescriptor &other) = delete;
			ConvolutionDescriptor(ConvolutionDescriptor &&other) noexcept = delete;
			ConvolutionDescriptor& operator=(const ConvolutionDescriptor &other) = delete;
			ConvolutionDescriptor& operator=(ConvolutionDescriptor &&other) noexcept = delete;
			~ConvolutionDescriptor() noexcept
			{
				cudnnStatus_t status = cudnnDestroyConvolutionDescriptor(m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
			size_t getWorkspaceSize() const noexcept
			{
				return m_workspace_size;
			}
			cudnnConvolutionFwdAlgo_t getAlgorithm() const noexcept
			{
				return m_algo;
			}
			operator cudnnConvolutionDescriptor_t() const noexcept
			{
				return m_desc;
			}
	};

}

namespace ml
{
	void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const mlShape_t output_shape = make_shape( { input_shape.dim[0], input_shape.dim[1], input_shape.dim[2], weights_shape.dim[0] });

		const TensorDescriptor xDesc(input_shape, dtype);
		const FilterDescriptor wDesc(weights_shape, dtype);
		const TensorDescriptor yDesc(output_shape, dtype);

		assert(weights_shape.dim[1] == weights_shape.dim[2]); // square kernel
		const mlDataType_t compute_type = cuda::has_fp16_math(context) ? DTYPE_FLOAT16 : DTYPE_FLOAT32;
		const ConvolutionDescriptor convDesc(cuda::Context::getCudnnHandle(context), xDesc, wDesc, yDesc, compute_type, weights_shape.dim[1]);

		cuda::Context::setWorkspaceSize(context, convDesc.getWorkspaceSize());
		cudnnHandle_t handle = cuda::Context::getCudnnHandle(context);
		const size_t workspace_size = cuda::Context::getWorkspaceSize(context);
		void *workspace = cuda::Context::getWorkspace(context);

		const float alpha1 = 1.0f;
		if (bias == nullptr)
		{
			const float beta = (add == nullptr) ? 0.0f : 1.0f;
			if (add != nullptr)
				cuda_memcpy_within_device(context, output, 0, add, size_of(dtype) * volume(output_shape));

			cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha1, xDesc, input, wDesc, weights, convDesc, convDesc.getAlgorithm(),
					workspace, workspace_size, &beta, yDesc, output);
			assert(status == CUDNN_STATUS_SUCCESS);

			ml::cuda_activation_forward(context, dtype, output_shape, output, output, act);
		}
		else
		{
			const mlShape_t bias_shape = make_shape( { 1, 1, 1, weights_shape.dim[0] });
			const TensorDescriptor biasDesc(bias_shape, dtype);
			const TensorDescriptor zDesc(output_shape, dtype);

			const ActivationDescriptor activationDesc(act);

			const void *zData = (add == nullptr) ? output : add;
			const float alpha2 = (add == nullptr) ? 0.0f : 1.0f;
			cudnnStatus_t status = cudnnConvolutionBiasActivationForward(handle, &alpha1, xDesc, input, wDesc, weights, convDesc,
					convDesc.getAlgorithm(), workspace, workspace_size, &alpha2, zDesc, zData, biasDesc, bias, activationDesc, yDesc, output);
			assert(status == CUDNN_STATUS_SUCCESS);
		}
	}
}

#endif

