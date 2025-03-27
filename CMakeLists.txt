cmake_minimum_required(VERSION 3.21.0)
project(quda-kokkos-interface VERSION 0.1.0 LANGUAGES CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${CMAKE_CURRENT_BINARY_DIR}/kokkos)


find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# get QUDA
set(QUDA_HOME "" CACHE PATH "path to QUDA")
if("${QUDA_HOME}" STREQUAL "")
  message( FATAL_ERROR "QUDA_HOME must be defined" )
endif()
FIND_LIBRARY(QUDA_LIB quda ${QUDA_HOME}/lib)
include_directories(SYSTEM ${QUDA_HOME}/include)

add_executable(quda_plaq_test quda_plaq_test.cpp)
target_link_libraries(quda_plaq_test Kokkos::kokkos ${QUDA_LIB} ${MPI_CXX_LIBRARIES})
