/**
 * @file    ToyPasses.hpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Pass for MLIR lowering to LLVM-IR.
 */
#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <mlir/Pass/Pass.h>

#include "lowering.hpp"
#include "driver.hpp"

namespace mlir
{
    void registerToyToLLVMLoweringPass( toy::driverState* pst = nullptr );
    inline void registerToyPasses( toy::driverState* pst = nullptr )
    {
        registerToyToLLVMLoweringPass( pst );
    }
}    // namespace mlir

#define GEN_PASS_REGISTRATION
#include "ToyPasses.hpp.inc"

#endif    // TOY_PASSES_H

// vim: et ts=4 sw=4
