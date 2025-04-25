#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include "ToyToLLVMLowering.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
void registerToyPasses();
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "ToyPasses.h.inc"

#endif // TOY_PASSES_H
