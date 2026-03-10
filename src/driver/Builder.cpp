/// @file    Builder.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic subset of the MLIR builder for the silly language.
#include "Builder.hpp"
#include "DriverState.hpp"
#include "SourceManager.hpp"

namespace silly
{
    Builder::Builder( silly::SourceManager &s, const std::string &filename )
        : sm{s},
          driverState{ sm.getDriverState() },
          sourceFile{ filename },
          ctx{ sm.getContext() },
          builder( ctx ),
          rmod( mlir::ModuleOp::create( mlir::FileLineColLoc::get( builder.getStringAttr( filename ), 0, 0 ) ) ),
          mainIP{},
          currentFuncName{},
          functionStateMap{}
    {
        builder.setInsertionPointToStart( rmod->getBody() );
        typ.initialize( builder, ctx );
    }
}

// vim: et ts=4 sw=4
