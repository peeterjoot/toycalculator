/// @file    lowering.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Glue code for MLIR lowering to LLVM-IR.
#pragma once

#include <mlir/Pass/Pass.h>
#include "driver.hpp"

namespace mlir
{
    /// Glue state that allows for passing driverState to the lowering pass
    std::unique_ptr<Pass> createToyToLLVMLoweringPass( toy::driverState * pst = nullptr );
}    // namespace mlir

// vim: et ts=4 sw=4
