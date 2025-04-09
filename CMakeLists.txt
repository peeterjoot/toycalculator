cmake_minimum_required(VERSION 3.10)
project(Calculator)

if(EXISTS "/opt/rh/gcc-toolset-13/root/usr/bin/g++")
   set(CMAKE_CXX_COMPILER "/opt/rh/gcc-toolset-13/root/usr/bin/g++")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ToyCalculatorParser)

# my ubuntu/WSL system:
if(EXISTS "/usr/include/antlr4-runtime")
    include_directories(/usr/include/antlr4-runtime)
else()
    set(ANTLR4_PREFIX "/home/pjoot/antlr4/usr/local")
    include_directories(${ANTLR4_PREFIX}/include/antlr4-runtime)

    message(STATUS "ANTLR4_PREFIX in: ${ANTLR4_PREFIX}")
endif()

if(NOT EXISTS "/usr/include/antlr4-runtime")
    link_directories(${ANTLR4_PREFIX}/lib64)
endif()

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen for dialect
set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${TABLEGEN_OUTPUT_DIR})

set(DIALECT_TD_FILE ${CMAKE_SOURCE_DIR}/src/ToyCalculatorDialect.td)
set(DIALECT_H_INC ${TABLEGEN_OUTPUT_DIR}/ToyCalculatorDialect.h.inc)
set(DIALECT_CPP_INC ${TABLEGEN_OUTPUT_DIR}/ToyCalculatorDialect.cpp.inc)

add_custom_command(
  OUTPUT ${DIALECT_H_INC}
  COMMAND ${MLIR_TABLEGEN_EXE}
    -gen-op-decls
    -I ${MLIR_INCLUDE_DIRS}
    -o ${DIALECT_H_INC}
    ${DIALECT_TD_FILE}
  DEPENDS ${DIALECT_TD_FILE}
  COMMENT "Generating ToyCalculatorDialect.h.inc"
)

add_custom_command(
  OUTPUT ${DIALECT_CPP_INC}
  COMMAND ${MLIR_TABLEGEN_EXE}
    -gen-op-defs
    -I ${MLIR_INCLUDE_DIRS}
    -o ${DIALECT_CPP_INC}
    ${DIALECT_TD_FILE}
  DEPENDS ${DIALECT_TD_FILE}
  COMMENT "Generating ToyCalculatorDialect.cpp.inc"
)

set(TABLEGEN_OUTPUT ${DIALECT_H_INC} ${DIALECT_CPP_INC})

add_executable(toycalculator
  src/driver.cpp
  src/ToyCalculatorDialect.cpp
  ToyCalculatorParser/ToyCalculatorListener.cpp
  ToyCalculatorParser/ToyCalculatorParser.cpp
  ToyCalculatorParser/ToyCalculatorBaseListener.cpp
  ToyCalculatorParser/ToyCalculatorLexer.cpp
)

add_custom_target(ToyCalculatorDialectInc DEPENDS ${TABLEGEN_OUTPUT})
add_dependencies(toycalculator ToyCalculatorDialectInc)

target_compile_options(toycalculator PRIVATE -Wall -Wextra -Wno-comment -Wno-overloaded-virtual -Wno-unused-parameter)
#target_compile_options(toycalculator PRIVATE -O2)
target_compile_options(toycalculator PRIVATE -g)

target_link_libraries(toycalculator
  PRIVATE
  antlr4-runtime
  LLVMSupport
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRBuiltin
)
