/*
 * vector_macros.hpp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_VECTOR_MACROS_HPP_
#define VECTORS_VECTOR_MACROS_HPP_

//#define __AVX2__

#define NAMESPACE_AVX2 ns_avx2
#define NAMESPACE_AVX ns_avx
#define NAMESPACE_SSE41 ns_sse41
#define NAMESPACE_SSE2 ns_sse2
#define NAMESPACE_NO_SIMD ns_none

#if defined(COMPILE_COMMON_CODE) and not defined(SIMD_LEVEL)
#  define SIMD_LEVEL 0
#  define SIMD_NAMESPACE NAMESPACE_NO_SIMD
#endif

#ifndef SIMD_LEVEL
#  if defined(__AVX512VL__) and defined(__AVX512BW__) and defined(__AVX512DQ__)
#    define SIMD_LEVEL 10
#    define SIMD_NAMESPACE NAMESPACE_AVX2
#  elif defined(__AVX512F__) or defined(__AVX512__)
#    define SIMD_LEVEL 9
#    define SIMD_NAMESPACE NAMESPACE_AVX2
#  elif defined(__AVX2__)
#    define SIMD_LEVEL 8
#    define SIMD_NAMESPACE NAMESPACE_AVX2
#  elif defined(__AVX__)
#    define SIMD_LEVEL 7
#    define SIMD_NAMESPACE NAMESPACE_AVX
#  elif defined(__SSE4_2__)
#    define SIMD_LEVEL 6
#    define SIMD_NAMESPACE NAMESPACE_SSE41
#  elif defined(__SSE4_1__)
#    define SIMD_LEVEL 5
#    define SIMD_NAMESPACE NAMESPACE_SSE41
#  elif defined(__SSSE3__)
#    define SIMD_LEVEL 4
#    define SIMD_NAMESPACE NAMESPACE_SSE2
#  elif defined(__SSE3__)
#    define SIMD_LEVEL 3
#    define SIMD_NAMESPACE NAMESPACE_SSE2
#  elif defined(__SSE2__) or defined(__x86_64__)
#    define SIMD_LEVEL 2
#    define SIMD_NAMESPACE NAMESPACE_SSE2
#  elif defined(__SSE__ )
#    define SIMD_LEVEL 1
#    define SIMD_NAMESPACE NAMESPACE_NO_SIMD
#  elif defined(_M_IX86_FP)
#    define SIMD_LEVEL _M_IX86_FP
#    define SIMD_NAMESPACE NAMESPACE_NO_SIMD
#  else
#    define SIMD_LEVEL 0
#    define SIMD_NAMESPACE NAMESPACE_NO_SIMD
#  endif
#endif

/* These are intended for use in other source files */
#define SUPPORTS_AVX512 0
#define SUPPORTS_AVX2 (SIMD_LEVEL >= 8)
#define SUPPORTS_AVX (SIMD_LEVEL >= 7)
#define SUPPORTS_SSE41 (SIMD_LEVEL >= 5)
#define SUPPORTS_SSE2 (SIMD_LEVEL >= 2)

#define SUPPORTS_FMA defined(__FMA__)
#define SUPPORTS_FP16 defined(__F16C__) or defined(__AVX512F__)
#define SUPPORTS_F16C defined(__F16C__)
#define SUPPORTS_BF16 defined(__AVX512BF16__) and defined(__AVX512F__)

/*
 *
 */
#define COMPILED_WITH_AVX512F (SIMD_LEVEL >= 9)
#define COMPILED_WITH_AVX2 (SIMD_LEVEL >= 8)
#define COMPILED_WITH_AVX (SIMD_LEVEL >= 7)
#define COMPILED_WITH_SSE41 (SIMD_LEVEL >= 5)
#define COMPILED_WITH_SSE2 (SIMD_LEVEL >= 2)

#define COMPILED_WITH_FMA defined(__FMA__)
#define COMPILED_WITH_F16C defined(__F16C__)

#endif /* VECTORS_VECTOR_MACROS_HPP_ */
