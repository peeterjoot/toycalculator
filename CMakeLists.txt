cmake_minimum_required(VERSION 3.16)
project(sillycompiler)

#------------------------------------------------------------------------------
# Install prefix configuration
option(SILLY_ENABLE_INSTALL "Enable installation of Silly compiler and samples" OFF)

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation prefix" FORCE)
endif()

option(USE_BISON_GRAMMAR "Use Bison/Flex grammar instead of ANTLR4" OFF)

#------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(USE_BISON_GRAMMAR)
    find_package(BISON 3.2 REQUIRED)
    find_package(FLEX REQUIRED)
else()
    set(ANTLR_OUTPUT_DIR "${CMAKE_BINARY_DIR}/src/antlr4Grammar/SillyParser" CACHE INTERNAL "")
    message(STATUS "ANTLR output directory: ${ANTLR_OUTPUT_DIR}")
endif()
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# What is find_package reporting:
message(STATUS "LLVM_FOUND .................... : ${LLVM_FOUND}")
message(STATUS "LLVM_VERSION .................. : ${LLVM_VERSION}")
message(STATUS "LLVM_PACKAGE_VERSION .......... : ${LLVM_PACKAGE_VERSION}")
#message(STATUS "LLVM_ROOT ..................... : ${LLVM_ROOT}")
message(STATUS "LLVM_BINARY_DIR ............... : ${LLVM_BINARY_DIR}")
message(STATUS "LLVM_LIBRARY_DIRS ............. : ${LLVM_LIBRARY_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS ............. : ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_DEFINITIONS .............. : ${LLVM_DEFINITIONS}")
message(STATUS "LLVM_ENABLE_RTTI .............. : ${LLVM_ENABLE_RTTI}")
message(STATUS "LLVM_ENABLE_EH ................ : ${LLVM_ENABLE_EH}")
message(STATUS "LLVM_TARGETS_TO_BUILD ......... : ${LLVM_TARGETS_TO_BUILD}")

message(STATUS "MLIR_FOUND .................... : ${MLIR_FOUND}")
message(STATUS "MLIR_INCLUDE_DIRS ............. : ${MLIR_INCLUDE_DIRS}")
#message(STATUS "MLIR_LIBRARY_DIRS ............. : ${MLIR_LIBRARY_DIRS}")
#message(STATUS "LLVM_CONFIG_CMAKE ............. : ${LLVM_CONFIG_CMAKE}")
#message(STATUS "MLIR_CONFIG_CMAKE ............. : ${MLIR_CONFIG_CMAKE}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

#------------------------------------------------------------------------------
# global include options for all components

message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/dialect)
include_directories(${CMAKE_SOURCE_DIR}/src/dialect)
include_directories(${CMAKE_SOURCE_DIR}/src/include)
if(USE_BISON_GRAMMAR)
    include_directories(/usr/include/antlr4-runtime)
endif()

#------------------------------------------------------------------------------
# compile options for all targets

function(add_common_compile_options target)
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -g
    )

    if(NOT LLVM_ENABLE_RTTI)
        target_compile_options(${target} PRIVATE
            -fno-rtti
        )
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("Compiler is g++")
        target_compile_options(${target} PRIVATE
          -fdiagnostics-show-option
        )

        # this is a nice option for understanding MLIR template overloads:
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14)
            message(STATUS "GCC >= 14 detected → adding -fdiagnostics-all-candidates")
            target_compile_options(${target} PRIVATE
                -fdiagnostics-all-candidates
            )
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("Compiler is clang++")
        # -Werror w/ g++-[13,14] is incompatible with llvm-project 22.1.0, with no
        # granular way of disabling various errors.  Restrict that to just when using
        # clang for now (these all used to be common gcc/clang compile options):
        target_compile_options(${target} PRIVATE
          -Werror
          -Wno-error=unused-but-set-variable
          -Wno-comment
          -Wno-overloaded-virtual
          -Wno-unused-parameter
        )

        # clang specific:
        target_compile_options(${target} PRIVATE
          -Wno-error=unused-private-field
          -Wno-unused-private-field
          -Wno-error=dangling-assignment-gsl
          -Wno-dangling-assignment-gsl
          -Wno-error=unused-variable
        )
    else()
        message("Compiler is neither g++ nor clang++: ${CMAKE_CXX_COMPILER_ID}")
    endif()
endfunction()

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
    install_silly_samples("tests/array" "array")
    install_silly_samples("tests/bool" "bool")
    install_silly_samples("tests/debug" "debug")
    install_silly_samples("tests/driver" "driver")
    install_silly_samples("tests/exit" "exit")
    install_silly_samples("tests/expressions" "expressions")
    install_silly_samples("tests/fatal" "fatal")
    install_silly_samples("tests/for" "for")
    install_silly_samples("tests/function" "function")
    install_silly_samples("tests/get" "get")
    install_silly_samples("tests/ifelifelse" "ifelifelse")
    install_silly_samples("tests/init" "init")
    install_silly_samples("tests/misc" "misc")
    install_silly_samples("tests/operators" "operators")
    install_silly_samples("tests/print" "print")
    install_silly_samples("tests/simple" "simple")
    install_silly_samples("tests/string" "string")
    install_silly_samples("tests/types" "types")

    # Install scripts
    install(PROGRAMS
        ${CMAKE_SOURCE_DIR}/bin/silly-opt
        DESTINATION bin
    )
endif()

