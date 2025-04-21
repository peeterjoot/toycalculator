#include "Toy_Dialect.h"

using namespace toy;

Toy_Dialect::Toy_Dialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<Toy_Dialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Toy_Dialect.cpp.inc"
  >();
}

// Include generated dialect definitions
#include "Toy_DialectDefs.cpp.inc"

// Include generated operation definitions
#define GET_OP_CLASSES
#include "Toy_Dialect.cpp.inc"

// vim: et ts=4 sw=4
