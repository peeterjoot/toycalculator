#include "ToyDialect.h"

#define GET_OP_CLASSES
#include "ToyDialect.cpp.inc"

// Include generated dialect definitions
#include "ToyDialectDefs.cpp.inc"

namespace toy {
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyDialect.cpp.inc"
      >();
}
} // namespace toy

// vim: et ts=4 sw=4
