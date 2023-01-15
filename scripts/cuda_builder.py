import os
import sys
from utils import get_platform, find_sources, run_commands, remove_ext


class CudaBuilder:
    def __init__(self, build_target: str, main_compiler: str, min_supported_arch: int):
        assert (build_target == 'release' or build_target == 'debug')
        assert (main_compiler == 'gcc' or main_compiler == 'msvc')
        self._build_target = build_target
        self._platform = get_platform()
        if self._platform == 'linux':
            assert (main_compiler == 'gcc')
        self._main_compiler = main_compiler
        self._min_sm_arch = min_supported_arch

        self._objects = []

    def _lib_name(self) -> str:
        if self._build_target == 'debug':
            tmp = 'cuda_math_d'
        else:
            tmp = 'cuda_math'
        if self._platform == 'linux':
            return tmp + '.a'
        else:
            if self._main_compiler == 'gcc':
                return tmp + '.dll'
            else:
                return tmp + '.lib'

    def _sm_archs(self) -> str:
        """

        :return:
        """
        sm_architectures = [10, 11, 12, 13, 20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86]
        return ' -arch=sm_' + str(self._min_sm_arch)  # TODO add proper multi-target compilation options

    def _compiler(self) -> str:
        """
        On Linux use GCC with C++ 11. On Windows use CL. If compiling for use with MSVC add /MD flags.

        :return:
        """
        if self._platform == 'linux':
            return 'nvcc -ccbin g++ -m64 -std=c++14'
        if self._platform == 'windows':
            tmp = 'nvcc -ccbin cl -m64'
            if self._main_compiler == 'msvc':
                if self._build_target == 'release':
                    tmp += ' -Xcompiler=\"/MD\"'
                else:
                    tmp += ' -Xcompiler=\"/MDd\"'
            return tmp

    def _optimizations(self) -> str:
        """
        For release use -O3 and turn off debug. For debug use -O0 and turn on device side debug.

        :return:
        """
        if self._build_target == 'release':
            return ' -O3 -DNDEBUG'
        else:
            return ' -O0 --device-debug'

    def _includes(self) -> str:
        """
        Append path to 'include' of libml.

        :return:
        """
        return ' -I\"../include/\" -I\"/usr/local/cuda/include/\"'

    def _misc_options(self) -> str:
        """
        In all cases append default stream options.
        On Windows if compiling for use with MINGW/GCC add flag to build DLL.

        :return:
        """
        tmp = ' -default-stream per-thread -DUSE_CUDA'
        if self._platform == 'windows':
            tmp += ' -DBUILDING_DLL'
        return tmp

    def _compile_file(self, path: str, filename: str) -> list:
        """
        Combine all parts of a compile command and use it to create object files saving it in ../bin/objects/.

        :param path: path to source file
        :param filename:  name of a source file
        :return: list of two produced object files
        """
        command = self._compiler() + self._sm_archs() + self._optimizations() + self._includes() + self._misc_options()

        tmp = remove_ext(filename)
       
        result = [command + ' -dc -o \"../bin/objects/cuda_' + tmp + '.o\" \"' + path + filename + '\"',
                  command + ' -dlink -o \"../bin/objects/cuda_' + tmp + '.dlink.o\" \"../bin/objects/cuda_' + tmp + '.o\"']
        self._objects += ['../bin/objects/cuda_' + tmp + '.o', '../bin/objects/cuda_' + tmp + '.dlink.o']
        return result

    def create_lib(self) -> list:
        """
        First run compilation of all source files. On Linux produce static library.
        On Windows when compiling for MINGW/GCC produce DLL, for MSVC produce static library.

        :return:
        """

        # first compile all source files
        self._objects = []
        list_of_sources = find_sources('../src/backend/cuda/', '.cu', '.cuh', []) + find_sources('../src/backend/cuda/', '.cpp', '.hpp', [])
        result = []
        result.append('export PATH=/usr/local/cuda/bin:$PATH')
        result.append('export CUDADIR=/usr/local/cuda/')
        result.append('export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH')
        for src in list_of_sources:
            result += self._compile_file(src[0], src[1])

        # then create command for library creation
        command = ''
        if self._main_compiler == 'gcc':
            if self._platform == 'linux':
                command = self._compiler() + self._sm_archs() + ' -lib -o'
            if self._platform == 'windows':
                command = self._compiler() + self._sm_archs() + ' -shared -o'
        if self._main_compiler == 'msvc':
            command = self._compiler() + self._sm_archs() + ' -lib -o'
        command += ' \"../bin/' + self._lib_name() + '\"'

        # append all compiled object files
        for o in self._objects:
            command += ' \"' + o + '\"'
        result.append(command)

        return result

    def get_objects(self) -> list:
        return self._objects

    def clear_objects(self) -> None:
        for o in self._objects:
            os.remove(o)
