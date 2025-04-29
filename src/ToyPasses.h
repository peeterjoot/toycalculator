#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include "ToyToLLVMLowering.h"
#include "mlir/Pass/Pass.h"

namespace mlir
{
    void registerToyToLLVMLoweringPass( bool isOptimized = false );
    inline void registerToyPasses( bool isOptimized = false )
    {
        registerToyToLLVMLoweringPass( isOptimized );
    }
}    // namespace mlir

#define GEN_PASS_REGISTRATION
#include "ToyPasses.h.inc"

#endif    // TOY_PASSES_H

// vim: et ts=4 sw=4
