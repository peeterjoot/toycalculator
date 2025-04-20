#include "Toy_Dialect.h"
#include "mlir/IR/OpImplementation.h"

namespace toy {

Toy_Dialect::Toy_Dialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<Toy_Dialect>()) {
    addOperations<
#define GET_OP_LIST
#include "Toy_Dialect.cpp.inc"
        >();
}

} // namespace toy

#include "Toy_Dialect.cpp.inc"

// vim: et ts=4 sw=4
