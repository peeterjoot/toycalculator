///
/// @file    SillyPasses.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Pass for MLIR lowering to LLVM-IR.
///
#pragma once

#include <mlir/Pass/Pass.h>

#include "driver.hpp"
#include "lowering.hpp"

namespace mlir
{
    void registerSillyToLLVMLoweringPass( silly::DriverState* pst = nullptr );

    /// Silly dialect pass framework
    inline void registerSillyPasses( silly::DriverState* pst = nullptr )
    {
        registerSillyToLLVMLoweringPass( pst );
    }

}    // namespace mlir

// vim: et ts=4 sw=4
