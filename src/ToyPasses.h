#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "ToyPasses.h.inc"
} // namespace mlir

#endif // TOY_PASSES_H
