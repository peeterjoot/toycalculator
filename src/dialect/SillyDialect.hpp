///
/// @file    SillyDialect.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Include the SillyDialect.md generated include files and their MLIR dependencies.
///
#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h> // NoMemoryEffect
#include <mlir/Support/TypeID.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/SymbolTable.h>

// Generated: dialect class forward declaration + getDialectNamespace()
#include "SillyDialectDecls.hpp.inc"

// Silly_VarType class forward/definition:
#include "SillyTypes.hpp"

#define GET_ATTRDEF_CLASSES
#include "SillyDialectEnums.h.inc"

// Generated: full op class definitions (AbortOp, DeclareOp, etc.)
// This must come AFTER Decls, before any .cpp.inc files
#define GET_OP_CLASSES
#include "SillyDialect.hpp.inc"

// vim: et ts=4 sw=4
