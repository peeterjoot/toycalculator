#ifndef TOY_TO_LLVM_LOWERING_H
#define TOY_TO_LLVM_LOWERING_H

#include <mlir/Pass/Pass.h>
#include "driver.h"

namespace mlir
{
    std::unique_ptr<Pass> createToyToLLVMLoweringPass( toy::driverState * pst = nullptr );
}    // namespace mlir

#endif    // TOY_TO_LLVM_LOWERING_H

// vim: et ts=4 sw=4
