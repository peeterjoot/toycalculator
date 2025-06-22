/**
 * @file    ToyPasses.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Glue code for the LLVM-IR lowering pass.
 */
#include "ToyPasses.hpp"
#include <mlir/Pass/PassManager.h>

namespace mlir
{
    void registerToyPasses()
    {
        ::registerToyPasses();    // Call the generated inline function
    }
}    // namespace mlir

// vim: et ts=4 sw=4
