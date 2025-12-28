/// @file    SillyPasses.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Pass for MLIR lowering to LLVM-IR.
#pragma once

#include <mlir/Pass/Pass.h>

#include "lowering.hpp"
#include "driver.hpp"

namespace mlir
{
    void registerSillyToLLVMLoweringPass( silly::DriverState* pst = nullptr );

    inline void registerSillyPasses( silly::DriverState* pst = nullptr )
    {
        registerSillyToLLVMLoweringPass( pst );
    }
}    // namespace mlir

#define GEN_PASS_REGISTRATION
#include "SillyPasses.hpp.inc"

// vim: et ts=4 sw=4
