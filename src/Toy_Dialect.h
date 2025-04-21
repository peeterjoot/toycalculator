#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"

namespace toy {
class Toy_Dialect : public mlir::Dialect {
public:
  explicit Toy_Dialect(mlir::MLIRContext *context);

};

// Include generated dialect declarations (includes getDialectNamespace)
#include "Toy_DialectDecls.h.inc"

} // namespace toy

// Include generated operation declarations
#define GET_OP_CLASSES
#include "Toy_Dialect.h.inc"

#endif // TOY_DIALECT_H

// vim: et ts=4 sw=4
