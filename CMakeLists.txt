cmake_minimum_required(VERSION 3.18)
project(MinML VERSION 1.0.0 DESCRIPTION "Minimalistic ML library" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_WITH_OPENBLAS "Use OpenBlas for linear algebra on CPU?" ON)
option(DYNAMIC_ARCH "Use runtime dispatching of simd code" OFF)
option(BUILD_WITH_CUDA "Build with CUDA support" ON)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type" FORCE)


add_library(MinML STATIC)
add_subdirectory("src/backend")
add_subdirectory("src/core")
add_subdirectory("src/graph")
add_subdirectory("src/layers")
add_subdirectory("src/training")
add_subdirectory("src/utils")

set_target_properties(MinML PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(MinML PUBLIC "${PROJECT_SOURCE_DIR}/include")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(MinML PUBLIC OpenMP::OpenMP_CXX)
else()
	message(FATAL_ERROR "no OpenMP")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_OPENBLAS)
	target_compile_definitions(MinML PRIVATE USE_OPENBLAS=1)
	target_link_libraries(MinML PUBLIC openblas)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_CUDA)
	target_compile_definitions(MinML PRIVATE USE_CUDA)
	target_include_directories(MinML PUBLIC "/usr/local/cuda/include")
	target_link_libraries(MinML PUBLIC cudart cublas)
endif()


