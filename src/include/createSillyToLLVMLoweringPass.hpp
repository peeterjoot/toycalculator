///
/// @file    createSillyToLLVMLoweringPass.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Glue code for MLIR lowering to LLVM-IR.
///
#pragma once

#include <mlir/Pass/Pass.h>

namespace silly
{
    class DriverState;
}

namespace mlir
{
    /// Glue state that allows for passing DriverState to the lowering pass
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass( silly::DriverState* pst = nullptr );
}    // namespace mlir

// vim: et ts=4 sw=4
