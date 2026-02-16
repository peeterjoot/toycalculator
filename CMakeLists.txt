cmake_minimum_required(VERSION 3.12)
project(sillycompiler)

#------------------------------------------------------------------------------
# Install prefix configuration
option(SILLY_ENABLE_INSTALL "Enable installation of Silly compiler and samples" OFF)

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation prefix" FORCE)
endif()

#------------------------------------------------------------------------------

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

add_subdirectory(tests)
add_subdirectory(src)

#------------------------------------------------------------------------------
if(SILLY_ENABLE_INSTALL)
    # Install documentation
    install(FILES
        ${CMAKE_SOURCE_DIR}/README.md
        DESTINATION share/silly/docs
    )

    file(GLOB CHANGELOG_FILES "${CMAKE_SOURCE_DIR}/Docs/Changelog*.md")

    install(FILES
        ${CHANGELOG_FILES}
        ${CMAKE_SOURCE_DIR}/Docs/TODO.md
        DESTINATION share/silly/docs
    )

    # Install sample programs
    # Helper function to install all .silly files from a directory
    function(install_silly_samples SOURCE_DIR DEST_SUBDIR)
        file(GLOB SILLY_FILES "${CMAKE_SOURCE_DIR}/${SOURCE_DIR}/*.silly")
        if(SILLY_FILES)
            install(FILES ${SILLY_FILES}
                DESTINATION share/silly/samples/${DEST_SUBDIR}
            )
        endif()
    endfunction()

    # Install samples from each category
    install_silly_samples("tests/endtoend/array" "array")
    install_silly_samples("tests/endtoend/bool" "bool")
    install_silly_samples("tests/endtoend/builtins" "builtins")
    install_silly_samples("tests/endtoend/debug" "debug")
    install_silly_samples("tests/endtoend/exit" "exit")
    install_silly_samples("tests/endtoend/expressions" "expressions")
    install_silly_samples("tests/endtoend/failure" "failure")
    install_silly_samples("tests/endtoend/for" "for")
    install_silly_samples("tests/endtoend/function" "function")
    install_silly_samples("tests/endtoend/ifelifelse" "ifelifelse")
    install_silly_samples("tests/endtoend/init" "init")
    install_silly_samples("tests/endtoend/misc" "misc")
    install_silly_samples("tests/endtoend/operators" "operators")
    install_silly_samples("tests/endtoend/print" "print")
    install_silly_samples("tests/endtoend/simple" "simple")
    install_silly_samples("tests/endtoend/string" "string")
    install_silly_samples("tests/endtoend/types" "types")

    # Install scripts
    install(PROGRAMS
        ${CMAKE_SOURCE_DIR}/bin/silly-opt
        DESTINATION bin
    )
endif()

