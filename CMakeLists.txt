cmake_minimum_required(VERSION 3.20)
project(Calculator LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM and MLIR
# these find_package's relied on:
#      -DLLVM_DIR=/usr/lib/llvm-19/cmake \
#      -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
#set(LLVM_DIR /usr/local/llvm-20/lib64/cmake/llvm)
#set(MLIR_DIR /usr/local/llvm-20/lib64/cmake/mlir)

# Include LLVM/MLIR directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Find ANTLR4 runtime
find_library(ANTLR4_RUNTIME NAMES antlr4-runtime libantlr4-runtime HINTS /usr/lib /usr/lib/x86_64-linux-gnu)
find_path(ANTLR4_INCLUDE_DIR NAMES antlr4-runtime.h HINTS /usr/include/antlr4-runtime)
if(NOT ANTLR4_RUNTIME)
  message(FATAL_ERROR "ANTLR4 runtime library not found. Please install libantlr4-runtime-dev.")
endif()
if(NOT ANTLR4_INCLUDE_DIR)
  message(FATAL_ERROR "ANTLR4 headers not found. Please install libantlr4-runtime-dev.")
endif()
message(STATUS "Found ANTLR4 runtime: ${ANTLR4_RUNTIME}")
message(STATUS "Found ANTLR4 include dir: ${ANTLR4_INCLUDE_DIR}")

# Define directories
set(GRAMMAR_DIR "${CMAKE_SOURCE_DIR}/grammar")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

# Include directories for generated files and ANTLR runtime
include_directories(${GRAMMAR_DIR})
include_directories(${ANTLR4_INCLUDE_DIR})

# Collect ANTLR-generated source files
file(GLOB GRAMMAR_SOURCES
  "${GRAMMAR_DIR}/*.cpp"
)

# Collect project source files
set(PROJECT_SOURCES
  "${SOURCE_DIR}/driver.cpp"
  "${SOURCE_DIR}/ToyCalculatorDialect.cpp"
)

# Define the executable
add_executable(calculator
  ${GRAMMAR_SOURCES}
  ${PROJECT_SOURCES}
)

# Link libraries
target_link_libraries(calculator
  PRIVATE
  ${ANTLR4_RUNTIME}
  LLVMCore
  LLVMSupport
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRParser
  MLIRTransforms
)

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})

# Optional: Enable warnings and optimizations
target_compile_options(calculator PRIVATE -Wall -Wextra -Wno-comment -Wno-overloaded-virtual -Wno-unused-parameter)
#target_compile_options(calculator PRIVATE -O2)
target_compile_options(calculator PRIVATE -g)

# Ensure MLIR/LLVM components are linked correctly
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(calculator PRIVATE ${LLVM_DEFINITIONS_LIST})
