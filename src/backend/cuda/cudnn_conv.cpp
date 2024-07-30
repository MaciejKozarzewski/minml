/*
 * cudnn_conv.cpp
 *
 *  Created on: Apr 7, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include <minml/backend/backend_utils.hpp>

#ifdef USE_CUDNN

#include "utils.hpp"

#include "cudnn_ops_infer.h"
#include "cudnn_cnn_infer.h"

#include <cassert>
#include <cstring>
#include <iostream>

namespace
{
	using namespace ml;
	cudnnDataType_t cudnn_convert(mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case DTYPE_UNKNOWN:
				return CUDNN_DATA_FLOAT;
			case DTYPE_FLOAT16:
				return CUDNN_DATA_HALF;
			case DTYPE_FLOAT32:
				return CUDNN_DATA_FLOAT;
			case DTYPE_INT32:
				return CUDNN_DATA_INT32;
		}
	}
	cudaDataType_t cuda_convert(mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case DTYPE_UNKNOWN:
				return CUDA_R_32F;
			case DTYPE_FLOAT16:
				return CUDA_R_16F;
			case DTYPE_FLOAT32:
				return CUDA_R_32F;
			case DTYPE_INT32:
				return CUDA_R_32I;
		}
	}
	cublasComputeType_t cublasLt_convert(mlDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case DTYPE_UNKNOWN:
				return CUBLAS_COMPUTE_32F;
			case DTYPE_FLOAT16:
				return CUBLAS_COMPUTE_16F;
			case DTYPE_FLOAT32:
				return CUBLAS_COMPUTE_32F;
			case DTYPE_INT32:
				return CUBLAS_COMPUTE_32F;
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
				status = cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NHWC, cudnn_convert(dtype), shape.dim[0], shape.dim[3], shape.dim[1],
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
				status = cudnnSetFilter4dDescriptor(m_desc, cudnn_convert(dtype), CUDNN_TENSOR_NHWC, shape.dim[0], shape.dim[3], shape.dim[1],
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
				status = cudnnSetActivationDescriptor(m_desc, convert(act), CUDNN_NOT_PROPAGATE_NAN, 0.0f);
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
				status = cudnnSetConvolution2dDescriptor(m_desc, padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, cudnn_convert(dtype));
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

	class MatrixLayout
	{
			cublasLtMatrixLayout_t m_layout = 0;
		public:
			MatrixLayout() noexcept = default;
			MatrixLayout(int rows, int cols, mlDataType_t dtype)
			{
				cublasStatus_t status = cublasLtMatrixLayoutCreate(&m_layout, cuda_convert(dtype), cols, rows, cols);
				assert(status == CUBLAS_STATUS_SUCCESS);
			}
			MatrixLayout(const mlShape_t &shape, mlDataType_t dtype) :
					MatrixLayout(shape.dim[0], shape.dim[1], dtype)
			{
				assert(shape.rank == 2);
			}
			MatrixLayout(const MatrixLayout &other) = delete;
			MatrixLayout(MatrixLayout &&other) noexcept = delete;
			MatrixLayout& operator=(const MatrixLayout &other) = delete;
			MatrixLayout& operator=(MatrixLayout &&other) noexcept = delete;
			~MatrixLayout() noexcept
			{
				cublasStatus_t status = cublasLtMatrixLayoutDestroy(m_layout);
				assert(status == CUBLAS_STATUS_SUCCESS);
			}
			operator cublasLtMatrixLayout_t() const noexcept
			{
				return m_layout;
			}
	};

	class MatMulDescriptor
	{
			cublasLtMatmulDesc_t m_desc = 0;
		public:
			MatMulDescriptor() noexcept = default;
			MatMulDescriptor(mlDataType_t dtype, const void *bias, mlActivationType_t act)
			{
				assert(act == ACTIVATION_LINEAR or ACTIVATION_RELU);
				cublasStatus_t status = cublasLtMatmulDescCreate(&m_desc, cublasLt_convert(dtype), cuda_convert(DTYPE_FLOAT32));
				assert(status == CUBLAS_STATUS_SUCCESS);

				int32_t tmp = CUBLAS_OP_T;
				status = cublasLtMatmulDescSetAttribute(m_desc, CUBLASLT_MATMUL_DESC_TRANSA, &tmp, sizeof(tmp));
				assert(status == CUBLAS_STATUS_SUCCESS);

				uint32_t epilogue;
				if (bias != nullptr)
				{
					status = cublasLtMatmulDescSetAttribute(m_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
					assert(status == CUBLAS_STATUS_SUCCESS);
					if (act == ACTIVATION_RELU)
						epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
					else
						epilogue = CUBLASLT_EPILOGUE_BIAS;
				}
				else
				{
					if (act == ACTIVATION_RELU)
						epilogue = CUBLASLT_EPILOGUE_RELU;
					else
						epilogue = CUBLASLT_EPILOGUE_DEFAULT;
				}

				status = cublasLtMatmulDescSetAttribute(m_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
				assert(status == CUBLAS_STATUS_SUCCESS);
			}
			MatMulDescriptor(const MatMulDescriptor &other) = delete;
			MatMulDescriptor(MatMulDescriptor &&other) noexcept = delete;
			MatMulDescriptor& operator=(const MatMulDescriptor &other) = delete;
			MatMulDescriptor& operator=(MatMulDescriptor &&other) noexcept = delete;
			~MatMulDescriptor() noexcept
			{
				cublasStatus_t status = cublasLtMatmulDescDestroy(m_desc);
				assert(status == CUBLAS_STATUS_SUCCESS);
			}
			operator cublasLtMatmulDesc_t() const noexcept
			{
				return m_desc;
			}
	};

	class TransformDescriptor
	{
			cublasLtMatrixTransformDesc_t m_desc = 0;
		public:
			TransformDescriptor()
			{
				cublasStatus_t status = cublasLtMatrixTransformDescCreate(&m_desc, cuda_convert(DTYPE_FLOAT32));
				assert(status == CUBLAS_STATUS_SUCCESS);
			}
			TransformDescriptor(const TransformDescriptor &other) = delete;
			TransformDescriptor(TransformDescriptor &&other) noexcept = delete;
			TransformDescriptor& operator=(const TransformDescriptor &other) = delete;
			TransformDescriptor& operator=(TransformDescriptor &&other) noexcept = delete;
			~TransformDescriptor() noexcept
			{
				cublasStatus_t status = cublasLtMatrixTransformDescDestroy(m_desc);
				assert(status == CUBLAS_STATUS_SUCCESS);
			}
			operator cublasLtMatrixTransformDesc_t() const noexcept
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

		assert(weights_shape.dim[1] == weights_shape.dim[2]); // square kernel

		const size_t workspace_size = cuda::Context::getWorkspaceSize(context);
		void *workspace = cuda::Context::getWorkspace(context);
		if (weights_shape.dim[1] == 1)
		{ // 1x1 convolution
			const MatrixLayout Adesc(volume_without_last_dim(input_shape), get_last_dim(input_shape), dtype);
			const MatrixLayout Bdesc(get_first_dim(weights_shape), volume_without_first_dim(weights_shape), dtype);
			const MatrixLayout Cdesc(volume_without_last_dim(output_shape), get_last_dim(output_shape), dtype);
			const MatrixLayout Ddesc(volume_without_last_dim(output_shape), get_last_dim(output_shape), dtype);
			const MatMulDescriptor computeDesc(dtype, bias, act);

			uint32_t alpha, beta;
			if (dtype == DTYPE_FLOAT32)
			{
				const float a = 1.0f;
				const float b = (add == nullptr) ? 0.0f : 1.0f;
				std::memcpy(&alpha, &a, sizeof(float));
				std::memcpy(&beta, &b, sizeof(float));
			}
			else
			{
				const half a = 1.0f;
				const half b = (add == nullptr) ? 0.0f : 1.0f;
				std::memcpy(&alpha, &a, sizeof(half));
				std::memcpy(&beta, &b, sizeof(half));
			}

			const void *C = (add == nullptr) ? output : add;

			cublasLtHandle_t handle = cuda::Context::getCublasLtHandle(context);
			cudaStream_t stream = cuda::Context::getStream(context);
			cublasStatus_t status = cublasLtMatmul(handle, computeDesc, &alpha, weights, Bdesc, input, Adesc, &beta, C, Cdesc, output, Ddesc, nullptr,
					workspace, workspace_size, stream);
			assert(status == CUBLAS_STATUS_SUCCESS);
		}
		else
		{
			const TensorDescriptor xDesc(input_shape, dtype);
			const FilterDescriptor wDesc(weights_shape, dtype);
			const TensorDescriptor yDesc(output_shape, dtype);

			const mlDataType_t compute_type = cuda::has_fp16_math(context) ? DTYPE_FLOAT16 : DTYPE_FLOAT32;
			const ConvolutionDescriptor convDesc(cuda::Context::getCudnnHandle(context), xDesc, wDesc, yDesc, compute_type, weights_shape.dim[1]);

			cuda::Context::setWorkspaceSize(context, convDesc.getWorkspaceSize());
			cudnnHandle_t handle = cuda::Context::getCudnnHandle(context);

			const float alpha1 = 1.0f;
			if (bias == nullptr)
			{
				const float beta = (add == nullptr) ? 0.0f : 1.0f;
				if (add != nullptr)
					cuda_memcpy_within_device(context, output, 0, add, 0, size_of(dtype) * volume(output_shape));

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

	void cuda_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, const void *bias, mlActivationType_t act)
	{
		const size_t workspace_size = cuda::Context::getWorkspaceSize(context);
		void *workspace = cuda::Context::getWorkspace(context);
		const MatrixLayout Adesc(shape_A, dtype);
		const MatrixLayout Bdesc(shape_B, dtype);
		const MatrixLayout Cdesc(shape_C, dtype);
		const MatrixLayout Ddesc(shape_D, dtype);
		const MatMulDescriptor computeDesc(dtype, bias, act);

		uint32_t _alpha, _beta;
		if (dtype == DTYPE_FLOAT32)
		{
			std::memcpy(&_alpha, &alpha, sizeof(float));
			std::memcpy(&_beta, &beta, sizeof(float));
		}
		else
		{
			const half a = static_cast<half>(alpha);
			const half b = static_cast<half>(beta);
			std::memcpy(&_alpha, &a, sizeof(half));
			std::memcpy(&_beta, &b, sizeof(half));
		}

		cublasLtHandle_t handle = cuda::Context::getCublasLtHandle(context);
		cudaStream_t stream = cuda::Context::getStream(context);
		cublasStatus_t status = cublasLtMatmul(handle, computeDesc, &_alpha, A, Bdesc, B, Adesc, &_beta, C, Cdesc, D, Ddesc, nullptr, workspace,
				workspace_size, stream);
		assert(status == CUBLAS_STATUS_SUCCESS);

		if (act != ACTIVATION_LINEAR and act != ACTIVATION_RELU)
			cuda_activation_forward(context, dtype, shape_D, D, D, act);
	}
}

#else
namespace ml
{
	void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
	}

	void cuda_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, const void *bias, mlActivationType_t act)
	{
		if (C != D)
			cuda_memcpy_within_device(context, D, 0, C, 0, volume(shape_D) * size_of(dtype));

		cuda_gemm(context, dtype, shape_D, D, shape_A, A, shape_B, B, opA, opB, alpha, beta);

		if (bias == nullptr)
			cuda_activation_forward(context, dtype, shape_D, D, D, act);
		else
			cuda_add_bias_act(context, dtype, shape_D, D, D, bias, act);
	}
}
#endif

