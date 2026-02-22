/// @file    DriverState.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   State to pass between driver and lowering pass.
#pragma once

#include <mlir/IR/Location.h>
#include <llvm/Passes/OptimizationLevel.h>

#include <cstdint>
#include <string>

/// Implicit function declaration for the body of a silly language program.
#define ENTRY_SYMBOL_NAME "main"

/// Name of the silly compiler driver, and used in llvm.ident and DICompileUnitAttr
#define COMPILER_NAME "silly"

namespace silly
{
    /// State to pass from the driver to CompilationUnit, parser/builder, lowering
    class DriverState
    {
       public:
        bool compileOnly{}; ///< -c
        bool assembleOnly{}; ///< -S
        bool keepTemps{}; ///< --keep-temp
        bool emitMLIR{}; ///< --emit-mlir
        bool emitMLIRBC{}; ///< --emit-mlirbc
        bool emitLLVM{}; ///< --emit-llvm
        bool toStdout{}; ///< --stdout
        bool noAbortPath{}; ///< --no-abort-path
        bool debugInfo{}; ///< True if -g is passed.
        bool verboseLink{}; ///< --verbose-link
        bool llvmDEBUG{}; ///< --debug-llvm
        bool noColorErrors{}; ///< --no-color-errors

        std::string outDir{}; ///< --output-directory
        std::string oName{}; ///< -o
        uint8_t initFillValue{}; ///< --init-fill value if specified (zero otherwise.)
        llvm::OptimizationLevel opt{}; ///< -O[0123], mapped from silly::OptLevel to llvm::OptimizationLevel

        /// Signal that -lm will be required when the program is linked (set by lowering)
        bool needsMathLib{};

        /// Signal that a source file could not be opened (set by parse-listener)
        bool openFailed{};

        /// Driver name
        const char * argv0{};

        /// &main for the driver
        void * mainSymbol{};
    };
}    // namespace silly

// vim: et ts=4 sw=4
