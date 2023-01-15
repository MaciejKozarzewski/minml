cmake_minimum_required(VERSION 3.18)
project(MinML VERSION 1.0.0 DESCRIPTION "Minimalistic ML library" LANGUAGES CXX CUDA)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_WITH_OPENBLAS "Use OpenBlas for linear algebra on CPU?" ON)
option(DYNAMIC_ARCH "Use runtime dispatching of simd code" OFF)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose build type" FORCE)


set(LIBRARY_NAME "MinML")
add_library(${LIBRARY_NAME} STATIC)
# add_subdirectory("backend")
add_subdirectory("core")
add_subdirectory("graph")
add_subdirectory("layers")
add_subdirectory("training")
add_subdirectory("utils")

set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
else()
	message(FATAL_ERROR "no OpenMP")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_OPENBLAS)
	target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_OPENBLAS=1)
	target_link_libraries(${LIBRARY_NAME} PUBLIC openblas)
endif()