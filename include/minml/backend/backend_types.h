/*
 * backend_types.h
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_BACKEND_BACKEND_TYPES_H_
#define MINML_BACKEND_BACKEND_TYPES_H_

namespace ml
{
#if defined(_WIN32) && defined(USE_CUDA)
#  ifdef IN_THE_DLL
#    define DLL_PUBLIC __declspec(dllexport)
#  else
#    define DLL_PUBLIC __declspec(dllimport)
#  endif
#else
#  define DLL_PUBLIC
#endif

#ifdef __cplusplus
	extern "C"
	{
#endif

		typedef struct
		{
				int rank;
				int dim[6];
		} mlShape_t;

		typedef struct
		{
				float scale;
				float shift;
		} mlQuantizationData_t;

		typedef enum
		{
			DTYPE_UNKNOWN,
			DTYPE_FLOAT8,
			DTYPE_FLOAT16,
			DTYPE_FLOAT32,
			DTYPE_FLOAT64,
			DTYPE_UINT8,
			DTYPE_INT8,
			DTYPE_INT16,
			DTYPE_INT32
		} mlDataType_t;

		typedef enum
		{
			ACTIVATION_LINEAR,
			ACTIVATION_SIGMOID,
			ACTIVATION_TANH,
			ACTIVATION_RELU,
			ACTIVATION_LEAKY_RELU,
			ACTIVATION_EXP
		} mlActivationType_t;

		typedef struct
		{
				void *data;
				mlDataType_t dtype;
				int rank;
				int dim[6];

		} mlTensor_t;

		typedef void *mlContext_t;
		typedef void *mlEvent_t;

#ifdef __cplusplus
	}
#endif
} /* namespace ml */

#endif /* MINML_BACKEND_BACKEND_TYPES_H_ */
