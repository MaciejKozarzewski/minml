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

#include "cudnn_ops.h"
#include "cudnn_cnn.h"

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
			case DTYPE_FLOAT64:
				return CUDNN_DATA_DOUBLE;
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
			case DTYPE_FLOAT64:
				return CUDA_R_64F;
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
			case DTYPE_FLOAT64:
				return CUBLAS_COMPUTE_64F;
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
		public:
			ConvolutionDescriptor() noexcept = default;
			ConvolutionDescriptor(cudnnHandle_t handle, const TensorDescriptor &xDesc, const FilterDescriptor &wDesc, const TensorDescriptor &yDesc,
					mlDataType_t dtype, int kernel_size, int group_count)
			{
				cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&m_desc);
				assert(status == CUDNN_STATUS_SUCCESS);
				status = cudnnSetConvolutionMathType(m_desc, CUDNN_TENSOR_OP_MATH);
				assert(status == CUDNN_STATUS_SUCCESS);
				const int padding = (kernel_size - 1) / 2;
				status = cudnnSetConvolution2dDescriptor(m_desc, padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, cudnn_convert(dtype));
				assert(status == CUDNN_STATUS_SUCCESS);

				if (group_count > 1)
				{
					status = cudnnSetConvolutionGroupCount(m_desc, group_count);
					assert(status == CUDNN_STATUS_SUCCESS);
				}
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
			operator cudnnConvolutionDescriptor_t() const noexcept
			{
				return m_desc;
			}
	};

	cudnnConvolutionFwdAlgoPerf_t get_conv_forward_algo(cudnnHandle_t handle, const TensorDescriptor &xDesc, const FilterDescriptor &wDesc,
			const TensorDescriptor &yDesc, const ConvolutionDescriptor &convDesc)
	{
		cudnnConvolutionFwdAlgoPerf_t result;
		int actual_count = 0;
		cudnnStatus_t status = cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, 1, &actual_count, &result);
		assert(status == CUDNN_STATUS_SUCCESS);
		assert(actual_count > 0);
		return result;
	}
	cudnnConvolutionBwdDataAlgoPerf_t get_conv_backward_data_algo(cudnnHandle_t handle, const TensorDescriptor &dxDesc, const FilterDescriptor &wDesc,
			const TensorDescriptor &dyDesc, const ConvolutionDescriptor &convDesc)
	{
		cudnnConvolutionBwdDataAlgoPerf_t result;
		int actual_count = 0;
		cudnnStatus_t status = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, wDesc, dyDesc, convDesc, dxDesc, 1, &actual_count, &result);
		assert(status == CUDNN_STATUS_SUCCESS);
		assert(actual_count > 0);
		return result;
	}
	cudnnConvolutionBwdFilterAlgoPerf_t get_conv_backward_filter_algo(cudnnHandle_t handle, const TensorDescriptor &xDesc,
			const FilterDescriptor &dwDesc, const TensorDescriptor &dyDesc, const ConvolutionDescriptor &convDesc)
	{
		cudnnConvolutionBwdFilterAlgoPerf_t result;
		int actual_count = 0;
		cudnnStatus_t status = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, xDesc, dyDesc, convDesc, dwDesc, 1, &actual_count, &result);
		assert(status == CUDNN_STATUS_SUCCESS);
		assert(actual_count > 0);
		return result;
	}

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

	void transpose_matrix(mlContext_t context, mlDataType_t dtype, void *output, const void *input, int rows, int columns)
	{
		cublasLtHandle_t handle = cuda::Context::getCublasLtHandle(context);
		cudaStream_t stream = cuda::Context::getStream(context);
		const float alpha = 1.0f;
		const float beta = 0.0f;

		const MatrixLayout Adesc(make_shape( { rows, columns }), dtype);
		const MatrixLayout Cdesc(make_shape( { columns, rows }), dtype);

		TransformDescriptor transformDesc;
		const cublasOperation_t op = CUBLAS_OP_T;
		cublasStatus_t status = cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op));
		assert(status == CUBLAS_STATUS_SUCCESS);

		status = cublasLtMatrixTransform(handle, transformDesc, &alpha, input, Adesc, &beta, nullptr, 0, output, Cdesc, stream);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

}

namespace ml
{
	void cuda_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		const mlShape_t output_shape = make_shape( { input_shape.dim[0], input_shape.dim[1], input_shape.dim[2], weights_shape.dim[0] });

		assert(weights_shape.dim[1] == weights_shape.dim[2]); // square kernel

		if (weights_shape.dim[1] == 1)
		{ // 1x1 convolution
			const MatrixLayout Adesc(volume_without_last_dim(input_shape), get_last_dim(input_shape), dtype);
			const MatrixLayout Bdesc(get_first_dim(weights_shape), volume_without_first_dim(weights_shape), dtype);
			const MatrixLayout Cdesc(volume_without_last_dim(output_shape), get_last_dim(output_shape), dtype);
			const MatrixLayout Ddesc(volume_without_last_dim(output_shape), get_last_dim(output_shape), dtype);
			const MatMulDescriptor computeDesc(dtype, bias, act);

			uint64_t alpha, beta;
			switch (dtype)
			{
				case DTYPE_FLOAT16:
				{
					const half a = 1.0f;
					const half b = (add == nullptr) ? 0.0f : 1.0f;
					std::memcpy(&alpha, &a, sizeof(half));
					std::memcpy(&beta, &b, sizeof(half));
					break;
				}
				case DTYPE_FLOAT32:
				{
					const float a = 1.0f;
					const float b = (add == nullptr) ? 0.0f : 1.0f;
					std::memcpy(&alpha, &a, sizeof(float));
					std::memcpy(&beta, &b, sizeof(float));
					break;
				}
				case DTYPE_FLOAT64:
				{
					const double a = 1.0;
					const double b = (add == nullptr) ? 0.0 : 1.0;
					std::memcpy(&alpha, &a, sizeof(double));
					std::memcpy(&beta, &b, sizeof(double));
					break;
				}
			}

			const void *C = (add == nullptr) ? output : add;

			cublasLtHandle_t handle = cuda::Context::getCublasLtHandle(context);
			cudaStream_t stream = cuda::Context::getStream(context);

			const size_t workspace_size = cuda::Context::getWorkspaceSize(context);
			void *workspace = cuda::Context::getWorkspace(context);

			cublasStatus_t status = cublasLtMatmul(handle, computeDesc, &alpha, weights, Bdesc, input, Adesc, &beta, C, Cdesc, output, Ddesc, nullptr,
					workspace, workspace_size, stream);
			assert(status == CUBLAS_STATUS_SUCCESS);
		}
		else
		{
			cudnnHandle_t handle = cuda::Context::getCudnnHandle(context);

			const TensorDescriptor xDesc(input_shape, dtype);
			const FilterDescriptor wDesc(weights_shape, dtype);
			const TensorDescriptor yDesc(output_shape, dtype);

			const ConvolutionDescriptor convDesc(cuda::Context::getCudnnHandle(context), xDesc, wDesc, yDesc, dtype, weights_shape.dim[1], 1);

			const cudnnConvolutionFwdAlgoPerfStruct perf = get_conv_forward_algo(handle, xDesc, wDesc, yDesc, convDesc);

			cuda::Context::setWorkspaceSize(context, perf.memory);
			const size_t workspace_size = cuda::Context::getWorkspaceSize(context);
			void *workspace = cuda::Context::getWorkspace(context);

			const float alpha1 = 1.0f;
			if (bias == nullptr)
			{
				const float beta = (add == nullptr) ? 0.0f : 1.0f;
				if (add != nullptr)
					cuda_memcpy_within_device(context, output, 0, add, 0, size_of(dtype) * volume(output_shape));

				cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha1, xDesc, input, wDesc, weights, convDesc, perf.algo, workspace,
						workspace_size, &beta, yDesc, output);
				assert(status == CUDNN_STATUS_SUCCESS);

				const mlTensor_t xy = ml::make_tensor(output, dtype, output_shape);
				ml::cuda_activation_forward(context, 1.0f, xy, 0.0f, xy, act);
			}
			else
			{
				const mlShape_t bias_shape = make_shape( { 1, 1, 1, weights_shape.dim[0] });
				const TensorDescriptor biasDesc(bias_shape, dtype);
				const TensorDescriptor zDesc(output_shape, dtype);

				const ActivationDescriptor activationDesc(act);

				const void *zData = (add == nullptr) ? output : add;
				const float alpha2 = (add == nullptr) ? 0.0f : 1.0f;
				cudnnStatus_t status = cudnnConvolutionBiasActivationForward(handle, &alpha1, xDesc, input, wDesc, weights, convDesc, perf.algo,
						workspace, workspace_size, &alpha2, zDesc, zData, biasDesc, bias, activationDesc, yDesc, output);
				assert(status == CUDNN_STATUS_SUCCESS);
			}
		}
	}

	void cuda_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, const void *bias, mlActivationType_t act)
	{
		if (is_transpose(opB) && !is_transpose(opA))
		{
			const size_t workspace_size = cuda::Context::getWorkspaceSize(context);
			void *workspace = cuda::Context::getWorkspace(context);
			const MatrixLayout Adesc(shape_A, dtype);
			const MatrixLayout Bdesc(shape_B, dtype);
			const MatrixLayout Cdesc(shape_C, dtype);
			const MatrixLayout Ddesc(shape_D, dtype);
			const MatMulDescriptor computeDesc(dtype, bias, act);

			uint64_t _alpha, _beta;
			switch (dtype)
			{
				case DTYPE_FLOAT16:
				{
					const half a = static_cast<half>(alpha);
					const half b = static_cast<half>(beta);
					std::memcpy(&_alpha, &a, sizeof(half));
					std::memcpy(&_beta, &b, sizeof(half));
					break;
				}
				case DTYPE_FLOAT32:
				{
					std::memcpy(&_alpha, &alpha, sizeof(float));
					std::memcpy(&_beta, &beta, sizeof(float));
					break;
				}
				case DTYPE_FLOAT64:
				{
					const double a = static_cast<double>(alpha);
					const double b = static_cast<double>(beta);
					std::memcpy(&_alpha, &a, sizeof(double));
					std::memcpy(&_beta, &b, sizeof(double));
					break;
				}
			}

			cublasLtHandle_t handle = cuda::Context::getCublasLtHandle(context);
			cudaStream_t stream = cuda::Context::getStream(context);
			cublasStatus_t status = cublasLtMatmul(handle, computeDesc, &_alpha, B, Bdesc, A, Adesc, &_beta, C, Cdesc, D, Ddesc, nullptr, workspace,
					workspace_size, stream);
			assert(status == CUBLAS_STATUS_SUCCESS);

			if (act != ACTIVATION_LINEAR && act != ACTIVATION_RELU)
			{
				const mlTensor_t xy = ml::make_tensor(D, dtype, shape_D);
				ml::cuda_activation_forward(context, 1.0f, xy, 0.0f, xy, act);
			}
		}
		else
		{
			if (C != D && beta != 0.0f)
				cuda_memcpy_within_device(context, D, 0, C, 0, volume(shape_D) * size_of(dtype));

			cuda_gemm(context, dtype, shape_D, D, shape_A, A, shape_B, B, opA, opB, alpha, beta);

			if (bias == nullptr)
			{
				const mlTensor_t xy = ml::make_tensor(D, dtype, shape_D);
				ml::cuda_activation_forward(context, 1.0f, xy, 0.0f, xy, act);
			}
			else
			{
				const mlTensor_t xy = ml::make_tensor(D, dtype, shape_D);
				const mlTensor_t b = ml::make_tensor(bias, dtype, make_shape( { get_last_dim(shape_D) }));
				cuda_add_bias_act(context, 1.0f, xy, b, 0.0f, xy, act);
			}
		}
	}

	void cudnn_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, const void *bias, void *output)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filter_size = weights_shape.dim[0];
		const int channels = weights_shape.dim[2];
		assert(weights_shape.dim[0] == weights_shape.dim[1]);

		cudnnHandle_t handle = cuda::Context::getCudnnHandle(context);

		const mlShape_t output_shape = input_shape;

		const TensorDescriptor xDesc(input_shape, dtype);
		const FilterDescriptor wDesc(make_shape( { channels, filter_size, filter_size, 1 }), dtype);
		const TensorDescriptor yDesc(output_shape, dtype);

		const ConvolutionDescriptor convDesc(handle, xDesc, wDesc, yDesc, dtype, filter_size, channels);

		const cudnnConvolutionFwdAlgoPerfStruct perf = get_conv_forward_algo(handle, xDesc, wDesc, yDesc, convDesc);

		cuda::Context::setWorkspaceSize(context, perf.memory);
		const size_t transposed_weights_size = volume(weights_shape) * size_of(dtype);
		const size_t workspace_size = cuda::Context::getWorkspaceSize(context) - transposed_weights_size;
		void *transposed_weights = cuda::Context::getWorkspace(context);
		void *workspace = reinterpret_cast<uint8_t*>(transposed_weights) + transposed_weights_size;

		transpose_matrix(context, dtype, transposed_weights, weights, filter_size * filter_size, channels);

		const float alpha1 = 1.0f;
		if (bias == nullptr)
		{
			const float beta = 0.0f;
			cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha1, xDesc, input, wDesc, transposed_weights, convDesc, perf.algo, workspace,
					workspace_size, &beta, yDesc, output);
			assert(status == CUDNN_STATUS_SUCCESS);
		}
		else
		{
			const mlShape_t bias_shape = make_shape( { 1, 1, 1, channels });
			const TensorDescriptor biasDesc(bias_shape, dtype);
			const TensorDescriptor zDesc(output_shape, dtype);

			const ActivationDescriptor activationDesc(ACTIVATION_LINEAR);

			const void *zData = output;
			const float alpha2 = 0.0f;
			cudnnStatus_t status = cudnnConvolutionBiasActivationForward(handle, &alpha1, xDesc, input, wDesc, transposed_weights, convDesc,
					perf.algo, workspace, workspace_size, &alpha2, zDesc, zData, biasDesc, bias, activationDesc, yDesc, output);
			assert(status == CUDNN_STATUS_SUCCESS);
		}
	}
	void cudnn_depthwise_conv_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *gradient_next,
			const void *weights, void *gradient_prev)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filter_size = weights_shape.dim[0];
		const int channels = weights_shape.dim[2];
		assert(weights_shape.dim[0] == weights_shape.dim[1]);

		cudnnHandle_t handle = cuda::Context::getCudnnHandle(context);

		const mlShape_t output_shape = input_shape;

		const TensorDescriptor dxDesc(input_shape, DTYPE_FLOAT32);
		const FilterDescriptor wDesc(make_shape( { channels, filter_size, filter_size, 1 }), DTYPE_FLOAT32);
		const TensorDescriptor dyDesc(output_shape, DTYPE_FLOAT32);

		const mlDataType_t compute_type = DTYPE_FLOAT32;
		const ConvolutionDescriptor convDesc(handle, dxDesc, wDesc, dyDesc, compute_type, filter_size, channels);

		const cudnnConvolutionBwdDataAlgoPerfStruct perf = get_conv_backward_data_algo(handle, dxDesc, wDesc, dyDesc, convDesc);

		cuda::Context::setWorkspaceSize(context, perf.memory);
		const size_t transposed_weights_size = volume(weights_shape) * size_of(DTYPE_FLOAT32);
		const size_t workspace_size = cuda::Context::getWorkspaceSize(context) - transposed_weights_size;
		void *transposed_weights = cuda::Context::getWorkspace(context);
		void *workspace = reinterpret_cast<uint8_t*>(transposed_weights) + transposed_weights_size;

		transpose_matrix(context, DTYPE_FLOAT32, transposed_weights, weights, filter_size * filter_size, channels);

		const float alpha = 1.0f;
		const float beta = 0.0f;

		cudnnStatus_t status = cudnnConvolutionBackwardData(handle, &alpha, wDesc, transposed_weights, dyDesc, gradient_next, convDesc, perf.algo,
				workspace, workspace_size, &beta, dxDesc, gradient_prev);
		assert(status == CUDNN_STATUS_SUCCESS);
	}
	void cudnn_depthwise_conv_update(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *gradient_next, void *weights_update)
	{
		assert(input_shape.rank == 4);
		assert(weights_shape.rank == 3);
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filter_size = weights_shape.dim[0];
		const int channels = weights_shape.dim[2];
		assert(weights_shape.dim[0] == weights_shape.dim[1]);

		cudnnHandle_t handle = cuda::Context::getCudnnHandle(context);

		const mlShape_t output_shape = input_shape;

		const TensorDescriptor xDesc(input_shape, DTYPE_FLOAT32);
		const FilterDescriptor dwDesc(make_shape( { channels, filter_size, filter_size, 1 }), DTYPE_FLOAT32);
		const TensorDescriptor dyDesc(output_shape, DTYPE_FLOAT32);

		const mlDataType_t compute_type = DTYPE_FLOAT32;
		const ConvolutionDescriptor convDesc(handle, xDesc, dwDesc, dyDesc, compute_type, filter_size, channels);

		const cudnnConvolutionBwdFilterAlgoPerfStruct perf = get_conv_backward_filter_algo(handle, xDesc, dwDesc, dyDesc, convDesc);

		cuda::Context::setWorkspaceSize(context, perf.memory);
		const size_t transposed_weights_size = volume(weights_shape) * size_of(DTYPE_FLOAT32);
		const size_t workspace_size = cuda::Context::getWorkspaceSize(context) - transposed_weights_size;
		void *transposed_weights = cuda::Context::getWorkspace(context);
		void *workspace = reinterpret_cast<uint8_t*>(transposed_weights) + transposed_weights_size;

		const float alpha = 1.0f;
		const float beta = 0.0f;

		cudnnStatus_t status = cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, input, dyDesc, gradient_next, convDesc, perf.algo, workspace,
				workspace_size, &beta, dwDesc, transposed_weights);
		assert(status == CUDNN_STATUS_SUCCESS);

		transpose_matrix(context, DTYPE_FLOAT32, weights_update, transposed_weights, channels, filter_size * filter_size);
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
		if (C != D && beta != 0.0f)
			cuda_memcpy_within_device(context, D, 0, C, 0, volume(shape_D) * size_of(dtype));

		cuda_gemm(context, dtype, shape_D, D, shape_A, A, shape_B, B, opA, opB, alpha, beta);

		if (bias == nullptr)
		{
			const mlTensor_t xy = ml::make_tensor(D, dtype, shape_D);
			ml::cuda_activation_forward(context, 1.0f, xy, 0.0f, xy, act);
		}
		else
		{
			const mlTensor_t xy = ml::make_tensor(D, dtype, shape_D);
			const mlTensor_t b = ml::make_tensor(bias, dtype, make_shape( { get_last_dim(shape_D) }));
			cuda_add_bias_act(context, 1.0f, xy, b, 0.0f, xy, act);
		}
	}

	void cudnn_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, const void *bias, void *output)
	{
	}
	void cudnn_depthwise_conv_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *gradient_next,
			const void *weights, void *gradient_prev)
	{
	}
	void cudnn_depthwise_conv_update(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *gradient_next, void *weights_update)
	{
	}

}
#endif

