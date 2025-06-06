cmake_minimum_required(VERSION 3.12)
project(toycalculator)

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

message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)
#include_directories(${CMAKE_BINARY_DIR}/ToyParser) # seems to be implied.

#------------------------------------------------------------------------------
# antlr4 related rules:
#
set(GRAMMAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Toy.g4")
set(ANTLR_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ToyParser")

# Ensure output directory exists
file(MAKE_DIRECTORY ${ANTLR_OUTPUT_DIR})

add_custom_command(
OUTPUT
    ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
    ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
    ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
    ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
COMMAND
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/runantlr --grammar Toy.g4 --outdir ${ANTLR_OUTPUT_DIR}
DEPENDS
    ${GRAMMAR_FILE}
    ${ANTLR4_LOCAL}
WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT
    "Generating ANTLR grammar files"
)

include_directories(/usr/include/antlr4-runtime)

# Create a custom target to ensure the grammar files are generated
add_custom_target(
    GenerateANTLR
    DEPENDS
        ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
        ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
        ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
        ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
)
#------------------------------------------------------------------------------
# mostly tablegen rules:
#

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen for dialect
set(LLVM_TARGET_DEFINITIONS src/calculator.td)

# Generate dialect declarations
mlir_tablegen(ToyDialectDecls.hpp.inc -gen-dialect-decls -dialect toy)

# Generate dialect definitions
mlir_tablegen(ToyDialectDefs.cpp.inc -gen-dialect-defs -dialect toy)

# Generate operation declarations
mlir_tablegen(ToyDialect.hpp.inc -gen-op-decls -dialect toy)

# Generate operation definitions
mlir_tablegen(ToyDialect.cpp.inc -gen-op-defs -dialect toy)

# Create a single TableGen target for all generated files
add_public_tablegen_target(ToyDialectIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialectDecls.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialectDefs.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.cpp.inc
)

# TableGen for passes
set(LLVM_TARGET_DEFINITIONS src/ToyPasses.td)
mlir_tablegen(ToyPasses.hpp.inc --gen-pass-decls -name Toy -I ${CMAKE_SOURCE_DIR}/src -I ${MLIR_INCLUDE_DIRS})

add_public_tablegen_target(ToyPassesIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/ToyPasses.hpp.inc
)
#------------------------------------------------------------------------------
# Targets:

# Add runtime library as a shared object
add_library(toy_runtime SHARED runtime/Toy_runtime.cpp)
set_target_properties(toy_runtime PROPERTIES OUTPUT_NAME "toy_runtime")
target_link_options(toy_runtime PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/runtime/version_script.map)

add_executable(toycalculator
  src/driver.cpp
  src/parser.cpp
  src/ToyDialect.cpp
  src/ToyPasses.cpp
  src/lowering.cpp
  ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
  ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
  ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
  ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
)

add_executable(mlirtest
  prototypes/mlirtest.cpp
)

add_executable(simplest
  prototypes/simplest.cpp
)

add_executable(hibye
  prototypes/hibye.cpp
)

#------------------------------------------------------------------------------
# compile options (and target dependencies)

# Include the output directory for header files
target_include_directories(toycalculator PRIVATE ${ANTLR_OUTPUT_DIR})

# Ensure the ANTLR and TableGen generation runs before building the project
add_dependencies(toycalculator hibye simplest mlirtest toy_runtime GenerateANTLR ToyDialectIncGen ToyPassesIncGen)

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

# -O2
# an attempt to suppress all the rhs == lhs ambiguous messages from the mlir headers:
#  -Wno-c++20-compat

target_compile_options(toy_runtime PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(toycalculator PRIVATE ${COMMON_COMPILE_OPTIONS})

#------------------------------------------------------------------------------
# Link rules for the toycalculator target:

# Check the target architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(IS_AARCH64 TRUE)
    set(LLVM_TARGET_ARCH "ARM")
else()
    set(IS_AARCH64 FALSE)
    set(LLVM_TARGET_ARCH "X86")
endif()

# Optional: Print the results
message(STATUS "Architecture is aarch64: ${IS_AARCH64}")
message(STATUS "LLVM Architecture specific libraries: ${LLVM_ARCH_LIBS}")

target_link_libraries(mlirtest
  PRIVATE
  LLVMSupport
  LLVMPasses
  LLVM${LLVM_TARGET_ARCH}AsmParser
  LLVM${LLVM_TARGET_ARCH}CodeGen
  LLVM${LLVM_TARGET_ARCH}Desc
  LLVM${LLVM_TARGET_ARCH}Info
  MLIRFuncToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRTransforms
  MLIRPass
  MLIRMemRefToLLVM
)

target_link_libraries(simplest
  PRIVATE
  LLVMSupport
  LLVMPasses
  LLVMMC
  LLVM${LLVM_TARGET_ARCH}AsmParser
  LLVM${LLVM_TARGET_ARCH}CodeGen
  LLVM${LLVM_TARGET_ARCH}Desc
  LLVM${LLVM_TARGET_ARCH}Info
  MLIRFuncToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRTransforms
  MLIRPass
  MLIRMemRefToLLVM
)

target_link_libraries(hibye
  PRIVATE
  LLVMSupport
  LLVMPasses
  LLVMMC
  LLVM${LLVM_TARGET_ARCH}AsmParser
  LLVM${LLVM_TARGET_ARCH}CodeGen
  LLVM${LLVM_TARGET_ARCH}Desc
  LLVM${LLVM_TARGET_ARCH}Info
  MLIRFuncToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRTransforms
  MLIRPass
  MLIRMemRefToLLVM
)

target_link_libraries(toycalculator
  PRIVATE
  antlr4-runtime
  LLVMSupport
  LLVMMC # llvm::TargetRegistry::lookupTarget
  LLVMAnalysis # llvm::AnalysisManager<llvm::LazyCallGraph::SCC, llvm::LazyCallGraph&>::AnalysisManager()
  LLVMPasses
  LLVM${LLVM_TARGET_ARCH}AsmParser
  LLVM${LLVM_TARGET_ARCH}CodeGen
  LLVM${LLVM_TARGET_ARCH}Desc
  LLVM${LLVM_TARGET_ARCH}Info
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRAsmParser
  MLIRTargetLLVMIRExport    # mlir::translateModuleToLLVMIR
  MLIRDataLayoutInterfaces
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
)

