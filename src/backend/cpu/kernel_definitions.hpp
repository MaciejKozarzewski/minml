/*
 * kernel_definitions.hpp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNEL_DEFINITIONS_HPP_
#define BACKEND_CPU_KERNEL_DEFINITIONS_HPP_

#include "vectors/vector_macros.hpp"
#include "utils.hpp"

#include <vector>

#if DYNAMIC_ARCH

namespace NAMESPACE_AVX2
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_AVX
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_SSE41
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_SSE2
{
#include "kernels/kernel_definitions.ipp"
}
namespace NAMESPACE_NO_SIMD
{
#include "kernels/kernel_definitions.ipp"
}

#else

namespace SIMD_NAMESPACE
{
#include "kernels/kernel_definitions.ipp"
}

#endif /* DYNAMIC_ARCH */


#endif /* BACKEND_CPU_KERNEL_DEFINITIONS_HPP_ */
