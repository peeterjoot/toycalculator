#include "ToyCalculatorDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace toy;

ToyDialect::ToyDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ToyDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "ToyCalculatorDialect.cpp.inc"
        >();
}

#include "ToyCalculatorDialect.cpp.inc"
