///
/// @file    SillyTypes.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   silly::varType custom printer
///

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include "SillyTypes.hpp"

namespace silly
{
    void varType::print( ::mlir::AsmPrinter &printer ) const
    {
        llvm::raw_ostream &os = printer.getStream();

        os << "<";
        printer.printType( getElementType() );

        auto shape = getShape().asArrayRef();
        if ( !shape.empty() )
        {
            os << "[";
            llvm::interleaveComma( shape, printer );
            os << "]";
        }

        os << ">";
    }

    ::mlir::Type varType::parse( ::mlir::AsmParser &parser )
    {
        // '<'
        if ( parser.parseLess() )
        {
            parser.emitError( parser.getCurrentLocation(), "expected '<'" );
            return Type();
        }

        // element type
        ::mlir::Type elementType;
        if ( parser.parseType( elementType ) )
        {
            parser.emitError( parser.getCurrentLocation(), "Failed to parse element type" );
            return ::mlir::Type();
        }

        // optional [ N ]
        ::llvm::SmallVector<int64_t, 1> dims;

        if ( succeeded( parser.parseOptionalLSquare() ) )
        {
            // exactly one integer expected
            int64_t size;
            if ( parser.parseInteger( size ) )
            {
                parser.emitError( parser.getCurrentLocation(), "array-size must be an integer" );
                return ::mlir::Type();
            }

            if ( size <= 0 )
            {
                parser.emitError( parser.getCurrentLocation(), "array size must be positive" );
                return ::mlir::Type();
            }

            dims.push_back( size );

            if ( parser.parseRSquare() )
            {
                parser.emitError( parser.getCurrentLocation(), "array-size must be followed immediately by ]" );
                return ::mlir::Type();
            }
        }

        // '>'
        if ( parser.parseGreater() )
        {
            parser.emitError( parser.getCurrentLocation(), "expected '>'" );
            return ::mlir::Type();
        }

        // build the type
        ::mlir::Builder &b = parser.getBuilder();
        ::mlir::MLIRContext *context = parser.getContext();
        auto shapeAttr = b.getDenseI64ArrayAttr( dims );

        return get( context, elementType, shapeAttr );
    }
}    // namespace silly

// vim: et ts=4 sw=4
