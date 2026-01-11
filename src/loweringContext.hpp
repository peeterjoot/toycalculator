/// @file loweringContext.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Helper class for lowering the MLIR silly dialect to the mlir LLVM dialect.

#pragma once

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>    // LLVMTypeConverter
#include <mlir/Dialect/Func/IR/FuncOps.h>                // mlir::func::FuncOp
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>             // mlir::LLVM::ConstantOp
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>               // LLVMPointerType, ...
#include <mlir/IR/Location.h>                            // FileLineColLoc

#include "SillyDialect.hpp"    // silly::CallOp, ...

namespace silly
{
    struct DriverState;

    /// Context object holding state and helper functions used during lowering
    /// of the Silly dialect to LLVM dialect.
    class LoweringContext
    {
       public:
        /// LLVM i1 type (cached for convenience).
        mlir::IntegerType tyI1;

        /// LLVM i8 type (cached for convenience).
        mlir::IntegerType tyI8;

        /// LLVM i16 type (cached for convenience).
        mlir::IntegerType tyI16;

        /// LLVM i32 type (cached for convenience).
        mlir::IntegerType tyI32;

        /// LLVM i64 type (cached for convenience).
        mlir::IntegerType tyI64;

        /// LLVM f32 type (cached for convenience).
        mlir::FloatType tyF32;

        /// LLVM f64 type (cached for convenience).
        mlir::FloatType tyF64;

        /// LLVM opaque pointer type (cached for convenience).
        mlir::LLVM::LLVMPointerType tyPtr;

        /// LLVM void type (cached for convenience).
        mlir::LLVM::LLVMVoidType tyVoid;

        /// Constructs the lowering context.
        /// @param moduleOp The module being lowered.
        /// @param driverState Reference to driver configuration state.
        LoweringContext( mlir::ModuleOp& moduleOp, const DriverState& driverState );

        /// Returns the preferred alignment for an element type according to the data layout.
        unsigned preferredTypeAlignment( mlir::Operation* op, mlir::Type elemType );

        /// Looks up the AllocaOp associated with a local variable name in the current function.
        mlir::LLVM::AllocaOp lookupLocalSymbolReference( mlir::Operation* op, const std::string& varName );

        /// Emits debug information for a local variable or array.
        void constructVariableDI( llvm::StringRef varName, mlir::Type& elemType, mlir::FileLineColLoc loc,
                                  unsigned elemSizeInBits, mlir::LLVM::AllocaOp& allocaOp, int64_t arraySize = 1 );

        /// Looks up or creates a global constant for a string literal.
        mlir::LLVM::GlobalOp lookupOrInsertGlobalOp( mlir::ConversionPatternRewriter& rewriter,
                                                     mlir::StringAttr& stringLit, mlir::Location loc, size_t strLen );

        /// Creates a call to the appropriate Silly print runtime function.
        silly::CallOp createPrintCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc,
                                       mlir::Value input, bool newline );

        void createAbortCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc );

        /// Creates a call to the appropriate Silly get runtime function.
        mlir::Value createGetCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc,
                                   mlir::Type inputType );

        /// Emits debug information for a function parameter.
        void constructParameterDI( mlir::FileLineColLoc loc, mlir::ConversionPatternRewriter& rewriter,
                                   const std::string& varName, mlir::LLVM::AllocaOp value, mlir::Type elemType,
                                   int paramIndex, const std::string& funcName );

        /// Casts a value to the target element type, performing necessary conversions.
        mlir::Value castToElemType( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, mlir::Value value,
                                    mlir::Type valType, mlir::Type elemType );

        /// Returns a reference to the internal LLVM type converter.
        inline mlir::LLVMTypeConverter& getTypeConverter();

        /// Checks if a type is floating-point (f32 or f64).
        inline bool isTypeFloat( mlir::Type ty ) const;

        /// Creates an i8 zero constant.
        inline mlir::LLVM::ConstantOp getI8zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Returns a zero constant for the given integer width (i8, i16, i32, i64).
        /// Throws an exception for unsupported widths.
        inline mlir::LLVM::ConstantOp getIzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                unsigned width );

        /// Creates a floating-point zero constant for the given width (32 or 64).
        mlir::LLVM::ConstantOp getFzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                         unsigned width );

        /// Creates an i64 one constant.
        inline mlir::LLVM::ConstantOp getI64one( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );


        /// Creates the DICompileUnit and basic debug types if debugging is enabled.
        void createDICompileUnit();

        /// Emits debug metadata for a function if debugging is enabled.
        void createFuncDebug( mlir::func::FuncOp funcOp );

        /// Associates an AllocaOp with a local variable name in the current function.
        void createLocalSymbolReference( mlir::LLVM::AllocaOp allocaOp, const std::string& varName );

        /// Looks up an existing global for a string literal.
        mlir::LLVM::GlobalOp lookupGlobalOp( mlir::StringAttr& stringLit );

       private:

        /// Returns the MLIR context.
        inline mlir::MLIRContext* getContext();

        /// Creates an i16 zero constant.
        inline mlir::LLVM::ConstantOp getI16zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Creates an i32 zero constant.
        inline mlir::LLVM::ConstantOp getI32zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Creates an i64 zero constant.
        inline mlir::LLVM::ConstantOp getI64zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Creates an f32 zero constant.
        inline mlir::LLVM::ConstantOp getF32zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Creates an f64 zero constant.
        inline mlir::LLVM::ConstantOp getF64zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Creates a prototype for a Silly print function if it does not exist.
        template <class Ty>
        void createSillyPrintPrototype( mlir::func::FuncOp& printOp, Ty type, const char* name );

        /// Creates the prototype for __silly_print_f64.
        inline void createSillyPrintF64Prototype();

        /// Creates the prototype for __silly_print_i64.
        inline void createSillyPrintI64Prototype();

        /// Creates the prototype for __silly_print_string.
        void createSillyPrintStringPrototype();

        /// Creates the prototype for __silly_abort.
        void createSillyAbortPrototype();

        /// Creates a prototype for a Silly get function if it does not exist.
        template <class RetTy>
        void createSillyGetPrototype( mlir::func::FuncOp& getOp, RetTy retType, const char* name );

        /// Creates the prototype for __silly_get_i1.
        inline void createSillyGetI1Prototype();

        /// Creates the prototype for __silly_get_i8.
        inline void createSillyGetI8Prototype();

        /// Creates the prototype for __silly_get_i16.
        inline void createSillyGetI16Prototype();

        /// Creates the prototype for __silly_get_i32.
        inline void createSillyGetI32Prototype();

        /// Creates the prototype for __silly_get_i64.
        inline void createSillyGetI64Prototype();

        /// Creates the prototype for __silly_get_f32.
        inline void createSillyGetF32Prototype();

        /// Creates the prototype for __silly_get_f64.
        inline void createSillyGetF64Prototype();

        /// Returns the corresponding LLVM debug type attribute for a given type.
        mlir::LLVM::DITypeAttr getDIType( mlir::Type type );

        /// Creates a DISubroutineType attribute for a function.
        mlir::LLVM::DISubroutineTypeAttr createDISubroutineType( mlir::func::FuncOp funcOp );

        /// Looks up the enclosing function name for an operation.
        std::string lookupFuncNameForOp( mlir::Operation* op );

        /// Debug file attribute (used when debugging is enabled).
        mlir::LLVM::DIFileAttr fileAttr;

        /// Map from function name to its DISubprogram attribute.
        std::unordered_map<std::string, mlir::LLVM::DISubprogramAttr> subprogramAttr;

        /// Map from string literal content to its GlobalOp.
        using StringLit2GlobalOp = std::unordered_map<std::string, mlir::LLVM::GlobalOp>;
        StringLit2GlobalOp stringLiterals;

        /// Prototype for __silly_print_f64.
        mlir::func::FuncOp printFuncF64;

        /// Prototype for __silly_print_i64.
        mlir::func::FuncOp printFuncI64;

        /// Prototype for __silly_print_string.
        mlir::func::FuncOp printFuncString;

        /// Prototype for __silly_abort.
        mlir::func::FuncOp printFuncAbort;

        /// Prototype for __silly_get_i1.
        mlir::func::FuncOp getFuncI1;

        /// Prototype for __silly_get_i8.
        mlir::func::FuncOp getFuncI8;

        /// Prototype for __silly_get_i16.
        mlir::func::FuncOp getFuncI16;

        /// Prototype for __silly_get_i32.
        mlir::func::FuncOp getFuncI32;

        /// Prototype for __silly_get_i64.
        mlir::func::FuncOp getFuncI64;

        /// Prototype for __silly_get_f32.
        mlir::func::FuncOp getFuncF32;

        /// Prototype for __silly_get_f64.
        mlir::func::FuncOp getFuncF64;

        /// Reference to driver configuration state.
        const DriverState& driverState;

        /// The module being lowered.
        mlir::ModuleOp& mod;

        /// OpBuilder positioned in the module.
        mlir::OpBuilder builder;

        /// Type converter for the lowering pass.
        mlir::LLVMTypeConverter typeConverter;

        /// Map from "funcName::varName" to AllocaOp for local variables.
        std::unordered_map<std::string, mlir::Operation*> symbolToAlloca;

        /// Debug compile unit attribute.
        mlir::LLVM::DICompileUnitAttr compileUnitAttr;

        /// Debug type for i8.
        mlir::LLVM::DITypeAttr diI8;

        /// Debug type for i16.
        mlir::LLVM::DITypeAttr diI16;

        /// Debug type for i32.
        mlir::LLVM::DITypeAttr diI32;

        /// Debug type for i64.
        mlir::LLVM::DITypeAttr diI64;

        /// Debug type for f32.
        mlir::LLVM::DITypeAttr diF32;

        /// Debug type for f64.
        mlir::LLVM::DITypeAttr diF64;

        /// Debug type for void.
        mlir::LLVM::DITypeAttr diVOID;

        /// Debug type for unknown/unsupported types.
        mlir::LLVM::DITypeAttr diUNKNOWN;
    };
}    // namespace silly

// vim: et ts=4 sw=4
