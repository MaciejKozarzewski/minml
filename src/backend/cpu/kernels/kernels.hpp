/*
 * kernels.hpp
 *
 *  Created on: Feb 10, 2025
 *      Author: maciek
 */

#ifndef BACKEND_CPU_KERNELS_KERNELS_HPP_
#define BACKEND_CPU_KERNELS_KERNELS_HPP_

#include <minml/backend/backend_types.h>

#include <cstddef>

namespace ml
{
	class TensorFragment;
}

namespace ml
{
	/*
	 * default kernels
	 */
	void average_pooling_def_1xN(const TensorFragment &input, TensorFragment &output) noexcept;
	void channel_scaling_def_1xN(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;

	/*
	 * sse2 kernels
	 */
	void average_pooling_sse2_1x32xfp32(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_sse2_1x16xfp64(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_sse2_1x4xfp32(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_sse2_1x2xfp64(const TensorFragment &input, TensorFragment &output) noexcept;

	void channel_scaling_sse2_1x32xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_sse2_1x16xfp64(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_sse2_1x4xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_sse2_1x2xfp64(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;

	/*
	 * avx kernels
	 */
	void convert_fp32_to_fp16_avx(void *dst, const void *src, size_t elements) noexcept;
	void convert_fp16_to_fp32_avx(void *dst, const void *src, size_t elements) noexcept;

	void average_pooling_avx_1x64xfp16(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_avx_1x64xfp32(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_avx_1x32xfp64(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_avx_1x8xfp16(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_avx_1x8xfp32(const TensorFragment &input, TensorFragment &output) noexcept;
	void average_pooling_avx_1x4xfp64(const TensorFragment &input, TensorFragment &output) noexcept;

	void channel_scaling_avx_1x64xfp16(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_avx_1x64xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_avx_1x32xfp64(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_avx_1x8xfp16(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_avx_1x8xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;
	void channel_scaling_avx_1x4xfp64(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept;

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_KERNELS_HPP_ */
