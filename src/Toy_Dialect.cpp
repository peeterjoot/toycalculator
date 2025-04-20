#include "Toy_Dialect.h"

#define GET_OP_CLASSES
#include "Toy_Dialect.cpp.inc"

using namespace toy;

Toy_Dialect::Toy_Dialect(mlir::MLIRContext *context)
    : Dialect("toy", context, mlir::TypeID::get<Toy_Dialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Toy_Dialect.cpp.inc"
      >();
}

// vim: et ts=4 sw=4
