/**
 * @file    stringtype.h
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   String type for the toy dialect.
 */
#if !defined __PJ_stringtype_h_is_included
#define __PJ_stringtype_h_is_included

#pragma once

#include <mlir/IR/Types.h>
#include <mlir/IR/TypeSupport.h>

namespace toy
{
    class StringType : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage>
    {
       public:
        using Base::Base;

        static StringType get( MLIRContext *context )
        {
            return Base::get( context );
        }
    };
}    // namespace toy

#endif
