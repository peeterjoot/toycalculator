///
/// @file DialectContext.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Registration glue for the silly dialect dependencies
///
#pragma once

#include <mlir/IR/MLIRContext.h>

namespace silly
{
    /// Context for MLIR dialect registration.
    struct DialectContext
    {
        /// MLIR context with loaded dialects.
        mlir::MLIRContext context;

        /// Loads required dialects (Silly, Func, Arith, MemRef, LLVM, SCF).
        DialectContext();
    };
}
