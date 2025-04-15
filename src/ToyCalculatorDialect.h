#ifndef TOY_CALCULATOR_DIALECT_H
#define TOY_CALCULATOR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace toy {

class ToyDialect : public mlir::Dialect {
public:
    ToyDialect(mlir::MLIRContext *context);
    static llvm::StringRef getDialectNamespace() { return "toy"; }
};

} // namespace toy

#define GET_OP_CLASSES
#include "ToyCalculatorDialect.h.inc"

#endif // TOY_CALCULATOR_DIALECT_H

// vim: et ts=4 sw=4
