///
/// @file    SillyDialect.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Includes the source headers generated from SillyDialect.td
///
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>    // mlir::func::FuncOp
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Types.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>

#include "SillyDialect.hpp"

// Pull in generated type method bodies (parse, print, etc. if any)
#define GET_TYPEDEF_CLASSES
#include "SillyTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "SillyDialectEnums.cpp.inc"

// Pull in generated op method bodies, adaptors, verify(), fold(), etc.
#define GET_OP_CLASSES
#include "SillyDialect.cpp.inc"

using namespace mlir;

namespace silly
{
    /// MLIR initialization boilerplate for the silly dialect
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

#if 0
    // not needed for now.  was work around for:
    //
    // << (() ? "static-string" : ty)
    //
    // which bombed due to type mismatch, not because Type isn't streamable
    //
    static llvm::SmallString<32> typeToString( mlir::Type type )
    {
        llvm::SmallString<32> buf;
        llvm::raw_svector_ostream os( buf );
        type.print( os );
        return buf;
    }
#endif

    /// Verifier for silly::DeclareOp
    mlir::LogicalResult DeclareOp::verify()
    {
        // Result type must be !silly.var<...>
        auto varType = mlir::dyn_cast<silly::varType>( getVar().getType() );
        if ( !varType )
        {
            // TODO: no coverage -- not sure if we can get here.  Think that the MLIR infra enforces this before the verify.
            // See: bad_declare_not_var_return.mlir
            return emitOpError( "result must be of type !silly.var" );
        }

        // Determine element count from the type shape
        llvm::ArrayRef<int64_t> shape = varType.getShape();

        if ( !shape.empty() )
        {
            if ( shape.size() != 1 )
            {
                // TODO: no coverage -- not sure if we can get here.  See: bad_declare_2darray.mlir -- type parser raises error first
                return emitOpError( "only 1D arrays are supported (rank must be 0 or 1)" );
            }
            if ( shape[0] <= 0 )
            {
                // TODO: no coverage -- see: bad_declare_negative_array_shape.mlir -- type parser raises error first.
                return emitOpError( "array size must be a positive integer" );
            }
        }

        // Check initializer count
        size_t numInits = getInitializers().size();
        size_t numElements = shape.empty() ? 1 : shape[0];

        if ( numInits > numElements )
        {
            // coverage: bad_declare_toomany_init_array.mlir bad_declare_toomany_init_scalar.mlir
            return emitOpError( "number of initializers (" )
                   << numInits << ") exceeds number of elements (" << numElements << ")";
        }

        // type-check initializers
        mlir::Type elemTy = varType.getElementType();
        for ( mlir::Value init : getInitializers() )
        {
            if ( init.getType() != elemTy )
            {
                // coverage: bad_dcl_init_mismatch_types.mlir
                return emitOpError( "initializer type " )
                       << init.getType() << " does not match variable element type " << elemTy;
            }
        }

        return mlir::success();
    }
}    // namespace silly

#include "SillyDialectDefs.cpp.inc"

/// Glue code for Silly dialect registration
extern "C" void registerSillyDialect( mlir::DialectRegistry &registry )
{
    registry.insert<silly::SillyDialect>();
}

/// Silly dialect plugin initialization
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo()
{
    return { /*.apiVersion =*/MLIR_PLUGIN_API_VERSION,
             /*.pluginName =*/"silly",
             /*.pluginVersion =*/"0.7",
             /*.registerDialects =*/[]( mlir::DialectRegistry *registry )
             { registry->insert<silly::SillyDialect>(); } };
}

// vim: et ts=4 sw=4
