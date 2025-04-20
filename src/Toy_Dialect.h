#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "Toy_Dialect.h.inc"    // For op declarations

namespace toy
{
    class Toy_Dialect : public mlir::Dialect
    {
       public:
        explicit Toy_Dialect( mlir::MLIRContext *context );
    };

}    // namespace toy

#endif    // TOY_DIALECT_H

// vim: et ts=4 sw=4
