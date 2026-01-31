///
/// @file    SillyDialect.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Includes the source headers generated from SillyDialect.td
///
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>

#include "SillyDialect.hpp"

// Pull in generated op method bodies, adaptors, verify(), fold(), etc.
#define GET_OP_CLASSES
#include "SillyDialect.cpp.inc"

// Pull in generated type method bodies (parse, print, etc. if any)
#define GET_TYPEDEF_CLASSES
#include "SillyTypes.cpp.inc"

using namespace mlir;

namespace silly
{

    void SillyDialect::initialize()
    {
        // Register types
        addTypes<
#define GET_TYPEDEF_LIST
#include "SillyTypes.cpp.inc"
            >();

        // Register operations
        addOperations<
#define GET_OP_LIST
#include "SillyDialect.cpp.inc"
            >();


        // llvm::errs() << "All registrations complete. Testing type print...\n";
    }
}    // namespace silly

#include "SillyDialectDefs.cpp.inc"

extern "C" void registerSillyDialect( mlir::DialectRegistry &registry )
{
    registry.insert<silly::SillyDialect>();
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo()
{
    return { /*.apiVersion =*/MLIR_PLUGIN_API_VERSION,
             /*.pluginName =*/"silly",
             /*.pluginVersion =*/"0.7",
             /*.registerDialects =*/[]( mlir::DialectRegistry *registry ) { registry->insert<silly::SillyDialect>(); } };
}

// vim: et ts=4 sw=4
