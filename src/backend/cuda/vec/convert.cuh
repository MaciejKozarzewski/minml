/*
 * convert.cuh
 *
 *  Created on: Jul 23, 2024
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_VEC_CONVERT_CUH_
#define BACKEND_CUDA_VEC_CONVERT_CUH_

#include "utils.cuh"
#include "vec4f.cuh"
#include "vec4h.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>

namespace vectors2
{

	HOST_DEVICE_INLINE vec4f convert_to_fp32(vec4h a)
	{
		return vec4f(__half2float(a.x0.x), __half2float(a.x0.y), __half2float(a.x1.x), __half2float(a.x1.y));
	}
	HOST_DEVICE_INLINE vec4h convert_to_fp16(vec4f a)
	{
		return vec4h(__float2half(a.x0), __float2half(a.x1), __float2half(a.x2), __float2half(a.x3));
	}

} /* namespace vectors2 */

#endif /* BACKEND_CUDA_VEC_CONVERT_CUH_ */
