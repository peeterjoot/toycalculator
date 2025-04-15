#include "ToyCalculatorDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OpDefinition.h"

using namespace toy;

void ToyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "ToyCalculatorDialect.cpp.inc"
        >();
}

ToyDialect::ToyDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ToyDialect>()) {
    initialize();
}

#include "ToyCalculatorDialect.cpp.inc"
