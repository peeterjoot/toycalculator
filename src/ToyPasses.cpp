#include "ToyPasses.h"
#include "ToyToLLVMLowering.h"
#include "mlir/Pass/PassManager.h"

#define GEN_PASS_REGISTRATION
#include "ToyPasses.h.inc"

void registerToyPasses() {
  registerPasses();
}
