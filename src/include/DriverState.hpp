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
    /// State to pass from the driver to parser/builder/lowering
    class DriverState
    {
       public:
        /// True if not OptLevel::O0
        bool isOptimized{};

        /// True if ABORT should omit path (--no-abort-path)
        bool abortOmitPath{};

        /// True if -g is passed.
        bool debugInfo{};

        /// True for color error messages (when output is a terminal.)
        bool colorErrors{};

        /// Numeric --init-fill value if specified (zero otherwise.)
        uint8_t fillValue{};

        bool emitMLIR{}; ///< --emit-mlir
        bool emitMLIRBC{}; ///< --emit-mlirbc
        bool emitLLVM{}; ///< --emit-llvm
        bool toStdout{}; ///< --stdout

        bool llvmDEBUG{}; ///< --debug-llvm
        std::string outDir{}; ///< --output-directory

        /// Signal that -lm will be required when the program is linked (set by lowering)
        bool needsMathLib{};

        /// Signal that the source file could not be opened (set by parse-listener)
        bool openFailed{};

        llvm::OptimizationLevel opt{}; ///< -O[0123]

        /// Driver name
        const char * argv0{};

        /// &main for the driver
        void * mainSymbol{};
    };
}    // namespace silly

// vim: et ts=4 sw=4
