cmake_minimum_required(VERSION 3.10)
project(toycalculator)

if(EXISTS "/opt/rh/gcc-toolset-13/root/usr/bin/g++")
   set(CMAKE_CXX_COMPILER "/opt/rh/gcc-toolset-13/root/usr/bin/g++")
endif()

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
include_directories(${CMAKE_SOURCE_DIR}/ToyParser)

# Define paths
set(ANTLR4_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/antlr-4.10-complete.jar")
#if(EXISTS /usr/bin/antlr4)
#    set(ANTLR4_SCRIPT "/usr/bin/antlr4")
#    set(ANTLR4_PREFIX "/usr")
#else()
#    set(ANTLR4_SCRIPT "${HOME}/.local/bin/antlr4")
#endif()
set(GRAMMAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Toy.g4")
set(ANTLR_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ToyParser")

# Ensure output directory exists
file(MAKE_DIRECTORY ${ANTLR_OUTPUT_DIR})

# Check if ~/.local/bin/antlr4 exists.
#if(EXISTS ${ANTLR4_SCRIPT})
#
#    message(STATUS "ANTLR4_PREFIX in: ${ANTLR4_PREFIX}")
#    include_directories(${ANTLR4_PREFIX}/include/antlr4-runtime)
#
#    # Use the local ANTLR4 script
#    message(STATUS "Using ANTLR4 from ${ANTLR4_SCRIPT}")
#    add_custom_command(
#        OUTPUT
#            ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
#            ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
#            ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
#            ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
#        COMMAND
#            ${ANTLR4_SCRIPT} -Dlanguage=Cpp ${GRAMMAR_FILE} -listener -o ${ANTLR_OUTPUT_DIR}
#        DEPENDS
#            ${GRAMMAR_FILE}
#        WORKING_DIRECTORY
#            ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT
#            "Generating ANTLR grammar files with ${ANTLR4_SCRIPT}"
#    )
#else()
    # Ubuntu/WSL, Fedora 42, or Ambian/ubuntu:

    # Use the Java-based ANTLR4 with the JAR file
    message(STATUS "Using ANTLR4 JAR from ${ANTLR4_LOCAL}")
    add_custom_command(
        OUTPUT
            ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
            ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
            ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
            ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
        COMMAND
            java -Xmx500M -cp ${ANTLR4_LOCAL} org.antlr.v4.Tool -Dlanguage=Cpp ${GRAMMAR_FILE} -listener -o ${ANTLR_OUTPUT_DIR}
        DEPENDS
            ${GRAMMAR_FILE}
            ${ANTLR4_LOCAL}
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT
            "Generating ANTLR grammar files with ANTLR4 JAR"
    )

    include_directories(/usr/include/antlr4-runtime)
#endif()

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

# Create a custom target to ensure the grammar files are generated
add_custom_target(
    GenerateANTLR
    DEPENDS
        ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
        ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
        ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
        ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
)

#if(NOT EXISTS "/usr/include/antlr4-runtime")
#    link_directories(${ANTLR4_PREFIX}/lib64)
#endif()

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen for dialect
set(LLVM_TARGET_DEFINITIONS src/ToyDialect.td)

# Generate dialect declarations
mlir_tablegen(ToyDialectDecls.h.inc -gen-dialect-decls -dialect toy)

# Generate dialect definitions
mlir_tablegen(ToyDialectDefs.cpp.inc -gen-dialect-defs -dialect toy)

# Generate operation declarations
mlir_tablegen(ToyDialect.h.inc -gen-op-decls -dialect toy)

# Generate operation definitions
mlir_tablegen(ToyDialect.cpp.inc -gen-op-defs -dialect toy)

# Create a single TableGen target for all generated files
add_public_tablegen_target(ToyDialectIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialectDecls.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialectDefs.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.cpp.inc
)

# TableGen for passes
set(LLVM_TARGET_DEFINITIONS src/ToyPasses.td)
mlir_tablegen(ToyPasses.h.inc --gen-pass-decls -name Toy -I ${CMAKE_SOURCE_DIR}/src -I ${MLIR_INCLUDE_DIRS})

add_public_tablegen_target(ToyPassesIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/ToyPasses.h.inc
)

# Add runtime library as a shared object
add_library(toy_runtime SHARED runtime/Toy_runtime.cpp)
set_target_properties(toy_runtime PROPERTIES OUTPUT_NAME "toy_runtime")

add_executable(toycalculator
  src/driver.cpp
  src/ToyParser.cpp
  src/ToyDialect.cpp
  src/ToyPasses.cpp
  src/ToyToLLVMLowering.cpp
  ${ANTLR_OUTPUT_DIR}/ToyBaseListener.cpp
  ${ANTLR_OUTPUT_DIR}/ToyListener.cpp
  ${ANTLR_OUTPUT_DIR}/ToyParser.cpp
  ${ANTLR_OUTPUT_DIR}/ToyLexer.cpp
)

# Include the output directory for header files
target_include_directories(toycalculator PRIVATE ${ANTLR_OUTPUT_DIR})

# Ensure the ANTLR and TableGen generation runs before building the project
add_dependencies(toycalculator toy_runtime GenerateANTLR ToyDialectIncGen ToyPassesIncGen)

target_compile_options(toycalculator PRIVATE
  -fdiagnostics-show-option
  -fdiagnostics-all-candidates
  -Wall
  -Wextra
  -Wno-comment
  -Wno-overloaded-virtual
  -Wno-unused-parameter
  -g
)
# an attempt to suppress all the rhs == lhs ambiguous messages from the mlir headers:
#  -Wno-c++20-compat

#target_compile_options(toycalculator PRIVATE -O2)

target_link_libraries(toycalculator
  PRIVATE
  antlr4-runtime
  LLVMSupport
  LLVM${LLVM_TARGET_ARCH}AsmParser
  LLVM${LLVM_TARGET_ARCH}CodeGen
  LLVM${LLVM_TARGET_ARCH}Desc
  LLVM${LLVM_TARGET_ARCH}Info
  MLIRAnalysis
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRCallInterfaces
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRControlFlowToLLVM
  MLIRDataLayoutInterfaces
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRFuncToLLVM
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLLVMDialect
  MLIRLoopLikeInterface
  MLIRMemorySlotInterfaces
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRParser
  MLIRPass
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRPDLToPDLInterp
  MLIRPresburger
  MLIRRewrite
  MLIRRewritePDL
  MLIRRuntimeVerifiableOpInterface
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSideEffectInterfaces
  MLIRSubsetOpInterface
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRUBDialect
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
  MLIRTargetLLVMIRExport    # mlir::translateModuleToLLVMIR
)

