#ifndef TOYDIALECT_H
#define TOYDIALECT_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/TypeID.h>

// Include generated dialect declarations (includes getDialectNamespace)
#include "ToyDialectDecls.h.inc"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "ToyDialect.h.inc"

#endif // TOYDIALECT_H

// vim: et ts=4 sw=4
