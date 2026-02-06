///
/// @file    SillyPasses.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Glue code for the LLVM-IR lowering pass.
///
#include "SillyPasses.hpp"

#include <mlir/Pass/PassManager.h>

// Generated pass registration function
#define GEN_PASS_REGISTRATION
#include "SillyPasses.hpp.inc"

namespace mlir
{
    /// Silly dialect pass glue code.
    void registerSillyPasses()
    {
        ::registerSillyPasses();    // Call the generated inline function
    }
}    // namespace mlir

// vim: et ts=4 sw=4
