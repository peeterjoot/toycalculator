cmake_minimum_required(VERSION 3.12)
project(sillycompiler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(ANTLR_OUTPUT_DIR "${CMAKE_BINARY_DIR}/src/grammar/SillyParser" CACHE INTERNAL "")

message(STATUS "ANTLR output directory: ${ANTLR_OUTPUT_DIR}")

#------------------------------------------------------------------------------
# global include options for all components

message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/dialect)
include_directories(${CMAKE_SOURCE_DIR}/src/dialect)
include_directories(${CMAKE_SOURCE_DIR}/src/include)
include_directories(/usr/include/antlr4-runtime)

#------------------------------------------------------------------------------
# compile options for all targets

set(COMMON_COMPILE_OPTIONS
  -Wall
  -Wextra
  -Werror
  -Wno-error=unused-but-set-variable
  -Wno-comment
  -Wno-overloaded-virtual
  -Wno-unused-parameter
  -g
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("Compiler is g++")
    list(APPEND COMMON_COMPILE_OPTIONS
      -fdiagnostics-show-option
      -fdiagnostics-all-candidates
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("Compiler is clang++")
    list(APPEND COMMON_COMPILE_OPTIONS
      -Wno-error=unused-private-field
      -Wno-unused-private-field
      -Wno-error=dangling-assignment-gsl
      -Wno-dangling-assignment-gsl
      -Wno-error=unused-variable
    )
else()
    message("Compiler is neither g++ nor clang++: ${CMAKE_CXX_COMPILER_ID}")
endif()

#------------------------------------------------------------------------------
enable_testing()

add_subdirectory(samples)
add_subdirectory(parsetests)
add_subdirectory(src)
#add_subdirectory(prototypes/standalone)
