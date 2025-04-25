#include "ToyPasses.h"
#include "mlir/Pass/PassManager.h"

namespace mlir
{
    void registerToyPasses()
    {
        ::registerToyPasses();    // Call the generated inline function
    }
}    // namespace mlir
