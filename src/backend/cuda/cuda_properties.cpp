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

		std::vector < std::string > result;
		for (auto prop = properties.begin(); prop < properties.end(); prop++)
		{
			std::string tmp = std::string(prop->name) + " : " + std::to_string(prop->multiProcessorCount) + " x ";
			tmp += "SM " + std::to_string(prop->major) + "." + std::to_string(prop->minor);
			tmp += " with " + std::to_string(prop->totalGlobalMem >> 20) + "MB of memory";
			result.push_back(tmp);
		}
		return result;
	}

	template<typename T>
	void print_field(const std::string &name, T x, const char *info = nullptr)
	{
		std::cout << name << " : " << std::to_string(x);
		if (info != nullptr)
			std::cout << "   (" << info << ")";
		std::cout << '\n';
	}

	std::string to_hex(uint8_t x)
	{
		static const std::array<char, 16> text ( { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' });
		return std::string(1, text[x / 16]) + std::string(1, text[x % 16]);
	}
	template<typename T>
	void print_hex(const std::string &name, T x, const char *info = nullptr)
	{
		std::cout << name << " : 0x";
		for (int i = 0; i < sizeof(T); i++)
			std::cout << ((i == 0) ? "" : ", ") << to_hex(reinterpret_cast<const uint8_t*>(&x)[i]);
		if (info != nullptr)
			std::cout << "   (" << info << ")";
		std::cout << '\n';
	}
	void print_text(const std::string &name, const char *x, const char *info = nullptr)
	{
		std::cout << name << " : \"" << x << "\"";
		if (info != nullptr)
			std::cout << "   (" << info << ")";
		std::cout << '\n';
	}
	template<typename T>
	void print_array(const std::string &name, const T *x, int len, const char *info = nullptr)
	{
		std::cout << name << " : [";
		for (int i = 0; i < len; i++)
			std::cout << ((i == 0) ? "" : ", ") << std::to_string(x[i]);
		std::cout << "]";
		if (info != nullptr)
			std::cout << "   (" << info << ")";
		std::cout << '\n';
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
	void cuda_print_device_features(int index)
	{
		if (index >= 0 && index < cuda_get_number_of_devices())
		{
			const cudaDeviceProp &prop = get_device_properties().at(index);
			print_text("name", prop.name, "ASCII string identifying device");
			print_hex("uuid", prop.uuid, "16-byte unique identifier");
			print_hex("luid", prop.luid, "8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms");
			print_hex("luidDeviceNodeMask", prop.luidDeviceNodeMask, "LUID device node mask. Value is undefined on TCC and non-Windows platforms");
			print_field("totalGlobalMem", prop.totalGlobalMem, "Global memory available on device in bytes");
			print_field("sharedMemPerBlock", prop.sharedMemPerBlock, "Shared memory available per block in bytes");
			print_field("regsPerBlock", prop.regsPerBlock, "32-bit registers available per bloc");
			print_field("warpSize", prop.warpSize, "Warp size in threads");
			print_field("memPitch", prop.memPitch, "Maximum pitch in bytes allowed by memory copies");
			print_field("maxThreadsPerBlock", prop.maxThreadsPerBlock, "Maximum number of threads per block");
			print_array("maxThreadsDim", prop.maxThreadsDim, 3, "Maximum size of each dimension of a block");
			print_array("maxGridSize", prop.maxGridSize, 3, "Maximum size of each dimension of a grid");
			print_field("clockRate", prop.clockRate, "Clock frequency in kilohertz");
			print_field("totalConstMem", prop.totalConstMem, "Constant memory available on device in bytes");
			print_field("major", prop.major, "Major compute capability");
			print_field("minor", prop.minor, "Minor compute capability");
			print_field("textureAlignment", prop.textureAlignment, "Alignment requirement for textures");
			print_field("texturePitchAlignment", prop.texturePitchAlignment,
					"Pitch alignment requirement for texture references bound to pitched memory");
			print_field("deviceOverlap", prop.deviceOverlap,
					"Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.");
			print_field("multiProcessorCount", prop.multiProcessorCount, "Number of multiprocessors on device");
			print_field("kernelExecTimeoutEnabled", prop.kernelExecTimeoutEnabled, "Specified whether there is a run time limit on kernels");
			print_field("integrated", prop.integrated, "Device is integrated as opposed to discrete");
			print_field("canMapHostMemory", prop.canMapHostMemory, "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer");
			print_field("computeMode", prop.computeMode, "Compute mode (See ::cudaComputeMode)");
			print_field("maxTexture1D", prop.maxTexture1D, "Maximum 1D texture size");
			print_field("maxTexture1DMipmap", prop.maxTexture1DMipmap, "Maximum 1D mipmapped texture size");
			print_field("maxTexture1DLinear", prop.maxTexture1DLinear,
					"Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.");
			print_array("maxTexture2D", prop.maxTexture2D, 2, " Maximum 2D texture dimensions");
			print_array("maxTexture2DMipmap", prop.maxTexture2DMipmap, 2, "Maximum 2D mipmapped texture dimensions");
			print_array("maxTexture2DLinear", prop.maxTexture2DLinear, 3,
					"Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory");
			print_array("maxTexture2DGather", prop.maxTexture2DGather, 2,
					"Maximum 2D texture dimensions if texture gather operations have to be performed");
			print_array("maxTexture3D", prop.maxTexture3D, 3, "Maximum 3D texture dimensions");
			print_array("maxTexture3DAlt", prop.maxTexture3DAlt, 3, "Maximum alternate 3D texture dimensions");
			print_field("maxTextureCubemap", prop.maxTextureCubemap, "Maximum Cubemap texture dimensions");
			print_array("maxTexture1DLayered", prop.maxTexture1DLayered, 2, "Maximum 1D layered texture dimensions");
			print_array("maxTexture2DLayered", prop.maxTexture2DLayered, 3, "Maximum 2D layered texture dimensions");
			print_array("maxTextureCubemapLayered", prop.maxTextureCubemapLayered, 2, "Maximum Cubemap layered texture dimensions");
			print_field("maxSurface1D", prop.maxSurface1D, "Maximum 1D surface size");
			print_array("maxSurface2D", prop.maxSurface2D, 2, "Maximum 2D surface dimensions");
			print_array("maxSurface3D", prop.maxSurface3D, 3, "Maximum 3D surface dimensions");
			print_array("maxSurface1DLayered", prop.maxSurface1DLayered, 2, "Maximum 1D layered surface dimensions");
			print_array("maxSurface2DLayered", prop.maxSurface2DLayered, 3, "Maximum 2D layered surface dimensions");
			print_field("maxSurfaceCubemap", prop.maxSurfaceCubemap, "Maximum Cubemap surface dimensions");
			print_array("maxSurfaceCubemapLayered", prop.maxSurfaceCubemapLayered, 2, " Maximum Cubemap layered surface dimensions");
			print_field("surfaceAlignment", prop.surfaceAlignment, "Alignment requirements for surfaces");
			print_field("concurrentKernels", prop.concurrentKernels, "Device can possibly execute multiple kernels concurrently");
			print_field("ECCEnabled", prop.ECCEnabled, "Device has ECC support enabled");
			print_field("pciBusID", prop.pciBusID, "PCI bus ID of the device");
			print_field("pciDeviceID", prop.pciDeviceID, "PCI device ID of the device");
			print_field("pciDomainID", prop.pciDomainID, "PCI domain ID of the device");
			print_field("tccDriver", prop.tccDriver, "1 if device is a Tesla device using TCC driver, 0 otherwise");
			print_field("asyncEngineCount", prop.asyncEngineCount, "Number of asynchronous engines");
			print_field("unifiedAddressing", prop.unifiedAddressing, "Device shares a unified address space with the host");
			print_field("memoryClockRate", prop.memoryClockRate, "Peak memory clock frequency in kilohertz");
			print_field("memoryBusWidth", prop.memoryBusWidth, "Global memory bus width in bits");
			print_field("l2CacheSize", prop.l2CacheSize, "Size of L2 cache in bytes");
			print_field("persistingL2CacheMaxSize", prop.persistingL2CacheMaxSize, "Device's maximum l2 persisting lines capacity setting in bytes");
			print_field("maxThreadsPerMultiProcessor", prop.maxThreadsPerMultiProcessor, "Maximum resident threads per multiprocessor");
			print_field("streamPrioritiesSupported", prop.streamPrioritiesSupported, "Device supports stream priorities");
			print_field("globalL1CacheSupported", prop.globalL1CacheSupported, "Device supports caching globals in L1");
			print_field("localL1CacheSupported", prop.localL1CacheSupported, "Device supports caching locals in L1");
			print_field("sharedMemPerMultiprocessor", prop.sharedMemPerMultiprocessor, "Shared memory available per multiprocessor in bytes");
			print_field("regsPerMultiprocessor", prop.regsPerMultiprocessor, "32-bit registers available per multiprocessor");
			print_field("managedMemory", prop.managedMemory, "Device supports allocating managed memory on this system");
			print_field("isMultiGpuBoard", prop.isMultiGpuBoard, "Device is on a multi-GPU board");
			print_field("multiGpuBoardGroupID", prop.multiGpuBoardGroupID, "Unique identifier for a group of devices on the same multi-GPU board");
			print_field("hostNativeAtomicSupported", prop.hostNativeAtomicSupported,
					"Link between the device and the host supports native atomic operations");
			print_field("singleToDoublePrecisionPerfRatio", prop.singleToDoublePrecisionPerfRatio,
					"Ratio of single precision performance (in floating-point operations per second) to double precision performance");
			print_field("pageableMemoryAccess", prop.pageableMemoryAccess,
					"Device supports coherently accessing pageable memory without calling cudaHostRegister on it");
			print_field("concurrentManagedAccess", prop.concurrentManagedAccess,
					"Device can coherently access managed memory concurrently with the CPU");
			print_field("computePreemptionSupported", prop.computePreemptionSupported, "Device supports Compute Preemption");
			print_field("canUseHostPointerForRegisteredMem", prop.canUseHostPointerForRegisteredMem,
					"Device can access host registered memory at the same virtual address as the CPU");
			print_field("cooperativeLaunch", prop.cooperativeLaunch,
					"Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel");
			print_field("cooperativeMultiDeviceLaunch", prop.cooperativeMultiDeviceLaunch,
					"Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.");
			print_field("sharedMemPerBlockOptin", prop.sharedMemPerBlockOptin, "Per device maximum shared memory per block usable by special opt in");
			print_field("pageableMemoryAccessUsesHostPageTables", prop.pageableMemoryAccessUsesHostPageTables,
					"Device accesses pageable memory via the host's page tables");
			print_field("directManagedMemAccessFromHost", prop.directManagedMemAccessFromHost,
					"Host can directly access managed memory on the device without migration.");
			print_field("maxBlocksPerMultiProcessor", prop.maxBlocksPerMultiProcessor, "Maximum number of resident blocks per multiprocessor");
			print_field("accessPolicyMaxWindowSize", prop.accessPolicyMaxWindowSize, "The maximum value of ::cudaAccessPolicyWindow::num_bytes.");
			print_field("reservedSharedMemPerBlock", prop.reservedSharedMemPerBlock, "Shared memory reserved by CUDA driver per block in bytes");
		}
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

	}
} /* namespace ml */

