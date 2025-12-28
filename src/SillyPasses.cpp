///
/// @file    SillyPasses.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Glue code for the LLVM-IR lowering pass.
///
#include "SillyPasses.hpp"

#include <mlir/Pass/PassManager.h>

namespace mlir
{
    void registerSillyPasses()
    {
        ::registerSillyPasses();    // Call the generated inline function
    }
}    // namespace mlir

// vim: et ts=4 sw=4
