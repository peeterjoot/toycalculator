#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"

namespace toy {

class Toy_Dialect : public mlir::Dialect {
public:
    Toy_Dialect(mlir::MLIRContext *context);
    static llvm::StringRef getDialectNamespace() { return "toy"; }
};

} // namespace toy

#define GET_OP_CLASSES
#include "Toy_Dialect.h.inc"

#endif // TOY_DIALECT_H

// vim: et ts=4 sw=4
