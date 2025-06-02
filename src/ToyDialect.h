/**
 * @file    ToyDialect.h
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Include the ToyDialect.md generated include files and their MLIR dependencies.
 */
#ifndef TOYDIALECT_H
#define TOYDIALECT_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/TypeID.h>
//#include <mlir/IR/SymbolTable.h> // this was for the experiment to add SymbolTable to ProgramOp.
// Trying to put a SymbolTable in ProgramOp was misguided, as ModuleOp already has one (as adding global string literals shows, because
// those need a SymbolTable -- you can see this by trying to add globalops without save/restore insertion point to the moduleop.)

// Include generated dialect declarations (includes getDialectNamespace)
#include "ToyDialectDecls.h.inc"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "ToyDialect.h.inc"

#endif // TOYDIALECT_H

// vim: et ts=4 sw=4
