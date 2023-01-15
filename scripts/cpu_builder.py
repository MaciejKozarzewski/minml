import os
import sys
from utils import get_platform, find_sources, run_commands, remove_ext, setup_msvc


class CpuBuilder:
    def __init__(self, build_target: str, main_compiler: str):
        assert (build_target == 'release' or build_target == 'debug')
        assert (main_compiler == 'gcc' or main_compiler == 'msvc')
        self._build_target = build_target
        self._platform = get_platform()
        if self._platform == 'linux':
            assert (main_compiler == 'gcc')
        self._main_compiler = main_compiler

        self._objects = []

        if self._main_compiler == 'gcc':
            self._simd_flags = {'sse_': '-msse', 'sse2_': '-msse2', 'sse3_': '-msse3', 'ssse3_': '-mssse3',
                                'sse4_': '-msse4', 'sse41_': '-msse4.1', 'sse42_': '-msse4.2', 'sse4a_': '-msse4a',
                                'avx_': '-mavx', 'avx2_': '-mavx2', 'avx512_': '-mavx512f'}
        else:
            self._simd_flags = {'sse_': '/arch:SSE', 'sse2_': '/arch:SSE2', 'sse3_': '/arch:AVX',
                                'ssse3_': '/arch:AVX', 'sse4_': '/arch:AVX', 'sse41_': '/arch:AVX',
                                'sse42_': '/arch:AVX', 'sse4a_': '/arch:AVX', 'avx_': '/arch:AVX',
                                'avx2_': '/arch:AVX2', 'avx512_': '/arch:AVX512'}

    def _lib_name(self) -> str:
        if self._build_target == 'debug':
            tmp = 'cpu_math_d'
        else:
            tmp = 'cpu_math'
        if self._main_compiler == 'gcc':
            return tmp + '.a'
        else:
            return tmp + '.lib'

    def _compiler(self) -> str:
        """
        On Linux use GCC. On Windows use CL.

        :return:
        """
        if self._main_compiler == 'gcc':
            return 'g++ -m64 -std=c++17 -fopenmp -c'
        if self._main_compiler == 'msvc':
            return 'cl /std:c++17 /c /EHsc'

    def _optimizations(self) -> str:
        """
        For release use -O3 and turn off debug. For debug use -O0 and turn on maximum debug level.

        :return:
        """
        if self._main_compiler == 'gcc':
            if self._build_target == 'release':
                return ' -O3 -DNDEBUG'
            else:
                return ' -O0 -g3'
        if self._main_compiler == 'msvc':
            if self._build_target == 'release':
                return ' /MD /O2 -DNDEBUG'
            else:
                return ' /MDd /Zi'

    def _includes(self) -> str:
        """
        Append path to 'include' of libml.
        :return:
        """
        if self._main_compiler == 'gcc':
            return ' -I\"../include/\"'
        if self._main_compiler == 'msvc':
            return ' /I\"../include/\"'

    def _misc_options(self) -> str:
        """
        :return:
        """
        return ''

    def _get_simd_flag(self, filename: str) -> str:
        for key in self._simd_flags:
            if filename.startswith(key):
                return ' ' + self._simd_flags[key]
        return ''

    def _compile_file(self, path: str, filename: str) -> list:
        """
        Combine all parts of a compile command and use it to create object files saving it in ../bin/objects/.

        :param path: path to source file
        :param filename:  name of a source file
        :return: list of two produced object files
        """

        command = self._compiler() + self._optimizations() + self._includes() + self._misc_options()
        command += self._get_simd_flag(filename)
        result = None
        if self._main_compiler == 'gcc':
            result = [command + ' -o \"../bin/objects/cpu_' + remove_ext(filename) + '.o\" \"' + path + filename + '\"']
        if self._main_compiler == 'msvc':
            result = [command + ' /Fo\"../bin/objects/cpu_' + remove_ext(filename) + '.o\" \"' + path + filename + '\"']
        assert (result is not None)
        self._objects += ['../bin/objects/cpu_' + remove_ext(filename) + '.o']
        return result

    def create_lib(self) -> list:
        """
        First run compilation of all source files to produce static library.

        :return:
        """

        # first compile all source files
        self._objects = []
        list_of_sources = find_sources('../src/backend/cpu/', '.cpp', '.hpp', [])
        result = []
        if self._main_compiler == 'msvc':
            result.append(setup_msvc())
        for src in list_of_sources:
            result += self._compile_file(src[0], src[1])

        # then create command for library creation
        command = ''
        if self._main_compiler == 'gcc':
            command = 'ar rcs -o '
        if self._main_compiler == 'msvc':
            command = 'lib /OUT:'
        command += '\"../bin/' + self._lib_name() + '\"'

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
