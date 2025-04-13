cmake_minimum_required(VERSION 3.20)
project(Calculator LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Include LLVM/MLIR directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Find ANTLR4 runtime
find_library(ANTLR4_RUNTIME antlr4-runtime REQUIRED)
if(NOT ANTLR4_RUNTIME)
  message(FATAL_ERROR "ANTLR4 runtime not found. Please install libantlr4-runtime or libantlr4-runtime-dev as appropriate.")
endif()

# Define directories
set(GRAMMAR_DIR "${CMAKE_SOURCE_DIR}/grammar")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

# Include directories for generated files and ANTLR runtime
include_directories(${GRAMMAR_DIR})
include_directories(${ANTLR4_INCLUDE_DIRS}) # Adjust if your ANTLR installation provides this

# Collect ANTLR-generated source files
file(GLOB GRAMMAR_SOURCES
  "${GRAMMAR_DIR}/*.cpp"
)

# Collect project source files
set(PROJECT_SOURCES
  "${SOURCE_DIR}/driver.cpp"
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
  MLIRParser
  MLIRSupport
)

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})

# Optional: Enable warnings and optimizations
target_compile_options(calculator PRIVATE -Wall -Wextra -O2)

# Ensure MLIR/LLVM components are linked correctly
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(calculator PRIVATE ${LLVM_DEFINITIONS_LIST})
