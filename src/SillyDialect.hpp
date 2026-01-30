/// @file    SillyDialect.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Include the SillyDialect.md generated include files and their MLIR dependencies.
#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/TypeID.h>

#include "SillyTypes.hpp"

// Include generated dialect declarations (includes getDialectNamespace)
#include "SillyDialectDecls.hpp.inc"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "SillyDialect.hpp.inc"

// vim: et ts=4 sw=4
