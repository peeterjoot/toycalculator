///
/// @file    helper.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Some generic MLIR helper functions
///
#include "helper.hpp"

#include <format>

namespace silly
{
    mlir::FileLineColLoc locationToFLCLoc( mlir::Location loc )
    {
        // Cast Location to FileLineColLoc
        mlir::FileLineColLoc fileLineLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        assert( fileLineLoc );

        return fileLineLoc;
    }

    mlir::func::FuncOp getEnclosingFuncOp( mlir::Operation* op )
    {
        while ( op )
        {
            if ( mlir::func::FuncOp funcOp = dyn_cast<mlir::func::FuncOp>( op ) )
            {
                return funcOp;
            }
            op = op->getParentOp();
        }
        return nullptr;
    }

    std::string lookupFuncNameForOp( mlir::Operation* op )
    {
        mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

        return funcOp.getSymName().str();
    }

    std::string mlirTypeToString( mlir::Type t )
    {
        std::string s;
        llvm::raw_string_ostream( s ) << t;
        return s;
    }

    std::string formatLocation( mlir::Location loc )
    {
        if ( mlir::FileLineColLoc fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(), fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

    std::string filenameFromLoc( mlir::Location loc )
    {
        if ( mlir::FileLineColLoc fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return fileLoc.getFilename().str();
        }
        return "";
    }

    mlir::Type biggestTypeOf( mlir::Type ty1, mlir::Type ty2 )
    {
        if ( ty1 == ty2 )
        {
            return ty1;
        }
        else if ( ty1.isF64() )
        {
            return ty1;
        }
        else if ( ty2.isF64() )
        {
            return ty2;
        }
        else if ( ty1.isF32() )
        {
            return ty1;
        }
        else if ( ty2.isF32() )
        {
            return ty2;
        }
        else
        {
            mlir::IntegerType ity1 = mlir::cast<mlir::IntegerType>( ty1 );
            mlir::IntegerType ity2 = mlir::cast<mlir::IntegerType>( ty2 );

            unsigned w1 = ity1.getWidth();
            unsigned w2 = ity2.getWidth();

            if ( w1 > w2 )
            {
                return ty1;
            }
            else
            {
                return ty2;
            }
        }
    }
}
