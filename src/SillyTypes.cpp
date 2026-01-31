///
/// @file    SillyTypes.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   silly::varType implementation
///
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>  // declares FieldParser
#include <mlir/IR/OpImplementation.h>       // defines specializations
#include <mlir/Support/LLVM.h>

#include "SillyTypes.hpp"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "SillyTypes.cpp.inc"

// vim: et ts=4 sw=4
