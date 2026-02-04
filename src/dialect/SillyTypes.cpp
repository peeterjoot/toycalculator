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
            // TODO: no coverage -- mlir parser raises this error first.  see: bad_declare_ty_no_less.mlir
            parser.emitError( parser.getCurrentLocation(), "expected '<'" );
            return Type();
        }

        // element type
        ::mlir::Type elementType;
        if ( parser.parseType( elementType ) )
        {
            // coverage: bad_declare_ty_no_etype.mlir
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
                // coverage: bad_var_type_abc.mlir
                parser.emitError( parser.getCurrentLocation(), "array-size must be an integer" );
                return ::mlir::Type();
            }

            if ( size <= 0 )
            {
                // coverage: bad_declare_negative_array_shape.mlir
                parser.emitError( parser.getCurrentLocation(), "array size must be positive" );
                return ::mlir::Type();
            }

            dims.push_back( size );

            if ( parser.parseRSquare() )
            {
                // coverage: bad_declare_2darray.mlir
                parser.emitError( parser.getCurrentLocation(), "array-size must be followed immediately by ]" );
                return ::mlir::Type();
            }
        }

        // '>'
        if ( parser.parseGreater() )
        {
            // TODO: no coverage -- mlir parser raises this error first. see: bad_declare_ty_no_gt.mlir
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
