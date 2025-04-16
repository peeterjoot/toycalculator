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

# Define paths
set(ANTLR4_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/antlr-4.10-complete.jar")
if(EXISTS /usr/bin/antlr4)
    set(ANTLR4_SCRIPT "/usr/bin/antlr4")
    set(ANTLR4_PREFIX "/usr")
else()
    set(ANTLR4_SCRIPT "${HOME}/.local/bin/antlr4")
endif()
set(GRAMMAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ToyCalculator.g4")
set(ANTLR_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ToyCalculatorParser")

# Ensure output directory exists
file(MAKE_DIRECTORY ${ANTLR_OUTPUT_DIR})

# Check if ~/.local/bin/antlr4 exists
if(EXISTS ${ANTLR4_SCRIPT})

    message(STATUS "ANTLR4_PREFIX in: ${ANTLR4_PREFIX}")
    include_directories(${ANTLR4_PREFIX}/include/antlr4-runtime)

    # Use the local ANTLR4 script
    message(STATUS "Using ANTLR4 from ${ANTLR4_SCRIPT}")
    add_custom_command(
        OUTPUT
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorBaseListener.cpp
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorListener.cpp
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorParser.cpp
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorLexer.cpp
        COMMAND
            ${ANTLR4_SCRIPT} -Dlanguage=Cpp ${GRAMMAR_FILE} -listener -o ${ANTLR_OUTPUT_DIR}
        DEPENDS
            ${GRAMMAR_FILE}
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT
            "Generating ANTLR grammar files with ${ANTLR4_SCRIPT}"
    )
else()
    # my ubuntu/WSL system:

    # Use the Java-based ANTLR4 with the JAR file
    message(STATUS "Using ANTLR4 JAR from ${ANTLR4_LOCAL}")
    add_custom_command(
        OUTPUT
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorBaseListener.cpp
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorListener.cpp
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorParser.cpp
            ${ANTLR_OUTPUT_DIR}/ToyCalculatorLexer.cpp
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
endif()

# Create a custom target to ensure the grammar files are generated
add_custom_target(
    GenerateANTLR
    DEPENDS
        ${ANTLR_OUTPUT_DIR}/ToyCalculatorBaseListener.cpp
        ${ANTLR_OUTPUT_DIR}/ToyCalculatorListener.cpp
        ${ANTLR_OUTPUT_DIR}/ToyCalculatorParser.cpp
        ${ANTLR_OUTPUT_DIR}/ToyCalculatorLexer.cpp
)

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
  ${ANTLR_OUTPUT_DIR}/ToyCalculatorBaseListener.cpp
  ${ANTLR_OUTPUT_DIR}/ToyCalculatorListener.cpp
  ${ANTLR_OUTPUT_DIR}/ToyCalculatorParser.cpp
  ${ANTLR_OUTPUT_DIR}/ToyCalculatorLexer.cpp
)

# Include the output directory for header files
target_include_directories(toycalculator PRIVATE ${ANTLR_OUTPUT_DIR})

# Ensure the ANTLR generation runs before building the project
add_dependencies(toycalculator GenerateANTLR)

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
