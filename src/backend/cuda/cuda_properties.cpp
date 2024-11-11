/*
 * cuda_properties.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>

#include "utils.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>

namespace
{
	template<typename T>
	void print_field(std::string &result, const std::string &name, T x, const char *info = nullptr)
	{
		result += name + " : " + std::to_string(x);
		if (info != nullptr)
			result += "   (" + std::string(info) + ")";
		result += '\n';
	}

	std::string to_hex(uint8_t x)
	{
		static const std::array<char, 16> text( { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' });
		return std::string(1, text[x / 16]) + std::string(1, text[x % 16]);
	}
	template<typename T>
	void print_hex(std::string &result, const std::string &name, T x, const char *info = nullptr)
	{
		result += name + " : 0x";
		for (int i = 0; i < sizeof(T); i++)
			result += ((i == 0) ? "" : ", ") + to_hex(reinterpret_cast<const uint8_t*>(&x)[i]);
		if (info != nullptr)
			result += "   (" + std::string(info) + ")";
		result += '\n';
	}
	void print_text(std::string &result, const std::string &name, const char *x, const char *info = nullptr)
	{
		result += name + " : \"" + x + "\"";
		if (info != nullptr)
			result += "   (" + std::string(info) + ")";
		result += '\n';
	}
	template<typename T>
	void print_array(std::string &result, const std::string &name, const T *x, int len, const char *info = nullptr)
	{
		result += name + " : [";
		for (int i = 0; i < len; i++)
			result += ((i == 0) ? "" : ", ") + std::to_string(x[i]);
		result += "]";
		if (info != nullptr)
			result += "   (" + std::string(info) + ")";
		result += '\n';
	}

	int get_number_of_devices()
	{
		static const int number = []()
		{
			int result = 0;
			cudaError_t status = cudaGetDeviceCount(&result);
			if (status != cudaSuccess)
				return 0;
			return result;
		}();
		return number;
	}
	const std::vector<cudaDeviceProp>& get_device_properties()
	{
		static const std::vector<cudaDeviceProp> properties = []()
		{
			std::vector<cudaDeviceProp> result;
			const int count = get_number_of_devices();
			for (int i = 0; i < count; i++)
			{
				cudaDeviceProp prop;
				cudaError_t status = cudaGetDeviceProperties(&prop, i);
				assert(status == cudaSuccess);
				result.push_back(prop);
			}
			return result;
		}();
		return properties;
	}
	std::vector<std::string> get_device_infos()
	{
		const std::vector<cudaDeviceProp> &properties = get_device_properties();

		std::vector<std::string> result;
		for (auto prop = properties.begin(); prop < properties.end(); prop++)
		{
			std::string tmp = std::string(prop->name) + " : " + std::to_string(prop->multiProcessorCount) + " x ";
			tmp += "SM " + std::to_string(prop->major) + "." + std::to_string(prop->minor);
			tmp += " with " + std::to_string(prop->totalGlobalMem >> 20) + "MB of memory";
			result.push_back(tmp);
		}
		return result;
	}
	std::vector<std::string> get_device_features()
	{
		const std::vector<cudaDeviceProp> &properties = get_device_properties();

		std::vector<std::string> result;
		for (auto prop = properties.begin(); prop < properties.end(); prop++)
		{
			std::string tmp;
			print_text(tmp, "name", prop->name, "ASCII string identifying device");
			print_hex(tmp, "uuid", prop->uuid, "16-byte unique identifier");
			print_hex(tmp, "luid", prop->luid, "8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms");
			print_hex(tmp, "luidDeviceNodeMask", prop->luidDeviceNodeMask,
					"LUID device node mask. Value is undefined on TCC and non-Windows platforms");
			print_field(tmp, "totalGlobalMem", prop->totalGlobalMem, "Global memory available on device in bytes");
			print_field(tmp, "sharedMemPerBlock", prop->sharedMemPerBlock, "Shared memory available per block in bytes");
			print_field(tmp, "regsPerBlock", prop->regsPerBlock, "32-bit registers available per bloc");
			print_field(tmp, "warpSize", prop->warpSize, "Warp size in threads");
			print_field(tmp, "memPitch", prop->memPitch, "Maximum pitch in bytes allowed by memory copies");
			print_field(tmp, "maxThreadsPerBlock", prop->maxThreadsPerBlock, "Maximum number of threads per block");
			print_array(tmp, "maxThreadsDim", prop->maxThreadsDim, 3, "Maximum size of each dimension of a block");
			print_array(tmp, "maxGridSize", prop->maxGridSize, 3, "Maximum size of each dimension of a grid");
			print_field(tmp, "clockRate", prop->clockRate, "Clock frequency in kilohertz");
			print_field(tmp, "totalConstMem", prop->totalConstMem, "Constant memory available on device in bytes");
			print_field(tmp, "major", prop->major, "Major compute capability");
			print_field(tmp, "minor", prop->minor, "Minor compute capability");
			print_field(tmp, "textureAlignment", prop->textureAlignment, "Alignment requirement for textures");
			print_field(tmp, "texturePitchAlignment", prop->texturePitchAlignment,
					"Pitch alignment requirement for texture references bound to pitched memory");
			print_field(tmp, "deviceOverlap", prop->deviceOverlap,
					"Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.");
			print_field(tmp, "multiProcessorCount", prop->multiProcessorCount, "Number of multiprocessors on device");
			print_field(tmp, "kernelExecTimeoutEnabled", prop->kernelExecTimeoutEnabled, "Specified whether there is a run time limit on kernels");
			print_field(tmp, "integrated", prop->integrated, "Device is integrated as opposed to discrete");
			print_field(tmp, "canMapHostMemory", prop->canMapHostMemory, "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer");
			print_field(tmp, "computeMode", prop->computeMode, "Compute mode (See ::cudaComputeMode)");
			print_field(tmp, "maxTexture1D", prop->maxTexture1D, "Maximum 1D texture size");
			print_field(tmp, "maxTexture1DMipmap", prop->maxTexture1DMipmap, "Maximum 1D mipmapped texture size");
			print_field(tmp, "maxTexture1DLinear", prop->maxTexture1DLinear,
					"Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.");
			print_array(tmp, "maxTexture2D", prop->maxTexture2D, 2, " Maximum 2D texture dimensions");
			print_array(tmp, "maxTexture2DMipmap", prop->maxTexture2DMipmap, 2, "Maximum 2D mipmapped texture dimensions");
			print_array(tmp, "maxTexture2DLinear", prop->maxTexture2DLinear, 3,
					"Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory");
			print_array(tmp, "maxTexture2DGather", prop->maxTexture2DGather, 2,
					"Maximum 2D texture dimensions if texture gather operations have to be performed");
			print_array(tmp, "maxTexture3D", prop->maxTexture3D, 3, "Maximum 3D texture dimensions");
			print_array(tmp, "maxTexture3DAlt", prop->maxTexture3DAlt, 3, "Maximum alternate 3D texture dimensions");
			print_field(tmp, "maxTextureCubemap", prop->maxTextureCubemap, "Maximum Cubemap texture dimensions");
			print_array(tmp, "maxTexture1DLayered", prop->maxTexture1DLayered, 2, "Maximum 1D layered texture dimensions");
			print_array(tmp, "maxTexture2DLayered", prop->maxTexture2DLayered, 3, "Maximum 2D layered texture dimensions");
			print_array(tmp, "maxTextureCubemapLayered", prop->maxTextureCubemapLayered, 2, "Maximum Cubemap layered texture dimensions");
			print_field(tmp, "maxSurface1D", prop->maxSurface1D, "Maximum 1D surface size");
			print_array(tmp, "maxSurface2D", prop->maxSurface2D, 2, "Maximum 2D surface dimensions");
			print_array(tmp, "maxSurface3D", prop->maxSurface3D, 3, "Maximum 3D surface dimensions");
			print_array(tmp, "maxSurface1DLayered", prop->maxSurface1DLayered, 2, "Maximum 1D layered surface dimensions");
			print_array(tmp, "maxSurface2DLayered", prop->maxSurface2DLayered, 3, "Maximum 2D layered surface dimensions");
			print_field(tmp, "maxSurfaceCubemap", prop->maxSurfaceCubemap, "Maximum Cubemap surface dimensions");
			print_array(tmp, "maxSurfaceCubemapLayered", prop->maxSurfaceCubemapLayered, 2, " Maximum Cubemap layered surface dimensions");
			print_field(tmp, "surfaceAlignment", prop->surfaceAlignment, "Alignment requirements for surfaces");
			print_field(tmp, "concurrentKernels", prop->concurrentKernels, "Device can possibly execute multiple kernels concurrently");
			print_field(tmp, "ECCEnabled", prop->ECCEnabled, "Device has ECC support enabled");
			print_field(tmp, "pciBusID", prop->pciBusID, "PCI bus ID of the device");
			print_field(tmp, "pciDeviceID", prop->pciDeviceID, "PCI device ID of the device");
			print_field(tmp, "pciDomainID", prop->pciDomainID, "PCI domain ID of the device");
			print_field(tmp, "tccDriver", prop->tccDriver, "1 if device is a Tesla device using TCC driver, 0 otherwise");
			print_field(tmp, "asyncEngineCount", prop->asyncEngineCount, "Number of asynchronous engines");
			print_field(tmp, "unifiedAddressing", prop->unifiedAddressing, "Device shares a unified address space with the host");
			print_field(tmp, "memoryClockRate", prop->memoryClockRate, "Peak memory clock frequency in kilohertz");
			print_field(tmp, "memoryBusWidth", prop->memoryBusWidth, "Global memory bus width in bits");
			print_field(tmp, "l2CacheSize", prop->l2CacheSize, "Size of L2 cache in bytes");
			print_field(tmp, "persistingL2CacheMaxSize", prop->persistingL2CacheMaxSize,
					"Device's maximum l2 persisting lines capacity setting in bytes");
			print_field(tmp, "maxThreadsPerMultiProcessor", prop->maxThreadsPerMultiProcessor, "Maximum resident threads per multiprocessor");
			print_field(tmp, "streamPrioritiesSupported", prop->streamPrioritiesSupported, "Device supports stream priorities");
			print_field(tmp, "globalL1CacheSupported", prop->globalL1CacheSupported, "Device supports caching globals in L1");
			print_field(tmp, "localL1CacheSupported", prop->localL1CacheSupported, "Device supports caching locals in L1");
			print_field(tmp, "sharedMemPerMultiprocessor", prop->sharedMemPerMultiprocessor, "Shared memory available per multiprocessor in bytes");
			print_field(tmp, "regsPerMultiprocessor", prop->regsPerMultiprocessor, "32-bit registers available per multiprocessor");
			print_field(tmp, "managedMemory", prop->managedMemory, "Device supports allocating managed memory on this system");
			print_field(tmp, "isMultiGpuBoard", prop->isMultiGpuBoard, "Device is on a multi-GPU board");
			print_field(tmp, "multiGpuBoardGroupID", prop->multiGpuBoardGroupID,
					"Unique identifier for a group of devices on the same multi-GPU board");
			print_field(tmp, "hostNativeAtomicSupported", prop->hostNativeAtomicSupported,
					"Link between the device and the host supports native atomic operations");
			print_field(tmp, "singleToDoublePrecisionPerfRatio", prop->singleToDoublePrecisionPerfRatio,
					"Ratio of single precision performance (in floating-point operations per second) to double precision performance");
			print_field(tmp, "pageableMemoryAccess", prop->pageableMemoryAccess,
					"Device supports coherently accessing pageable memory without calling cudaHostRegister on it");
			print_field(tmp, "concurrentManagedAccess", prop->concurrentManagedAccess,
					"Device can coherently access managed memory concurrently with the CPU");
			print_field(tmp, "computePreemptionSupported", prop->computePreemptionSupported, "Device supports Compute Preemption");
			print_field(tmp, "canUseHostPointerForRegisteredMem", prop->canUseHostPointerForRegisteredMem,
					"Device can access host registered memory at the same virtual address as the CPU");
			print_field(tmp, "cooperativeLaunch", prop->cooperativeLaunch,
					"Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel");
			print_field(tmp, "cooperativeMultiDeviceLaunch", prop->cooperativeMultiDeviceLaunch,
					"Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.");
			print_field(tmp, "sharedMemPerBlockOptin", prop->sharedMemPerBlockOptin,
					"Per device maximum shared memory per block usable by special opt in");
			print_field(tmp, "pageableMemoryAccessUsesHostPageTables", prop->pageableMemoryAccessUsesHostPageTables,
					"Device accesses pageable memory via the host's page tables");
			print_field(tmp, "directManagedMemAccessFromHost", prop->directManagedMemAccessFromHost,
					"Host can directly access managed memory on the device without migration.");
			print_field(tmp, "maxBlocksPerMultiProcessor", prop->maxBlocksPerMultiProcessor, "Maximum number of resident blocks per multiprocessor");
			print_field(tmp, "accessPolicyMaxWindowSize", prop->accessPolicyMaxWindowSize,
					"The maximum value of ::cudaAccessPolicyWindow::num_bytes.");
			print_field(tmp, "reservedSharedMemPerBlock", prop->reservedSharedMemPerBlock,
					"Shared memory reserved by CUDA driver per block in bytes");
			result.push_back(tmp);
		}
		return result;
	}

}

namespace ml
{
	int cuda_get_number_of_devices()
	{
		static const int result = get_number_of_devices();
		return result;
	}
	int cuda_get_memory(int index)
	{
		return get_device_properties().at(index).totalGlobalMem >> 20;
	}
	bool cuda_supports_type(int index, mlDataType_t dtype)
	{
		if (index >= 0 && index < cuda_get_number_of_devices())
		{
			switch (dtype)
			{
				case DTYPE_FLOAT16:
					return cuda::get_compute_capability(index) >= 53;
				case DTYPE_FLOAT32:
					return true;
			}
		}
		return false;
	}
	const char* cuda_get_device_info(int index)
	{
		static const std::vector<std::string> infos = get_device_infos();
		if (index >= 0 && index < cuda_get_number_of_devices())
			return infos.at(index).data();
		else
			return nullptr;
	}
	const char* cuda_get_device_features(int index)
	{
		static const std::vector<std::string> features = get_device_features();
		if (index >= 0 && index < cuda_get_number_of_devices())
			return features.at(index).data();
		else
			return nullptr;
	}
	void cuda_enable_tf32(mlContext_t context, bool b)
	{
		ml::cuda::Context::enableTF32(context, b);
	}

	namespace cuda
	{
		int get_compute_capability(int device_index)
		{
			return (get_device_properties()[device_index].major * 10 + get_device_properties()[device_index].minor);
		}

		bool has_fp16_math(mlContext_t context)
		{
			const int sm_ver = cuda::get_compute_capability(cuda::Context::getDeviceIndex(context));
			return sm_ver == 53 || sm_ver == 60 || sm_ver >= 62;
		}

		bool has_tensor_cores(mlContext_t context)
		{
			const int index = cuda::Context::getDeviceIndex(context);
			const int sm_ver = cuda::get_compute_capability(index);
			if (sm_ver >= 75)
			{
				if (sm_ver == 75)
				{
					const std::string name = get_device_properties().at(index).name;
					return name.find("RTX") != std::string::npos;
				}
				else
					return true;
			}
			else
				return false;

		}
		bool allows_tf32(mlContext_t context)
		{
			return cuda::Context::allowsTF32(context);
		}

	}
} /* namespace ml */

