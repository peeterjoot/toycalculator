#include "ToyCalculatorDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace toy {

ToyDialect::ToyDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ToyDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "ToyCalculatorDialect.cpp.inc"
        >();
}

} // namespace toy

#include "ToyCalculatorDialect.cpp.inc"

// vim: et ts=4 sw=4
