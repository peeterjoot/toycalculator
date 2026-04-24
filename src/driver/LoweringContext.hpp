///
/// @file LoweringContext.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Helper class for lowering the MLIR silly dialect to the mlir LLVM dialect.
///
#pragma once

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>    // LLVMTypeConverter
#include <mlir/Dialect/Func/IR/FuncOps.h>                // mlir::func::FuncOp
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>             // mlir::LLVM::ConstantOp
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>               // LLVMPointerType, ...
#include <mlir/IR/Location.h>                            // FileLineColLoc

#include "MlirTypeCache.hpp"
#include "PrintFlags.hpp"
#include "SillyDialect.hpp"

namespace silly
{
    class DriverState;

    /// Per-function lowering state
    struct PerFunctionLoweringState
    {
        /// Dwarf DI for that function, used to emit variable DI in lowering
        mlir::LLVM::DISubprogramAttr subProgramDI;

        /// The alloca op for PRINT arguments.
        ///
        /// This is sized big enough to hold the biggest number of arguments used for any PRINT statement in the
        /// function (all PRINT calls use this storage for their variable argument list.)
        mlir::LLVM::AllocaOp printArgs;

        /// Map from DeclareOp to AllocaOp for local variables.
        std::unordered_map<mlir::Operation*, mlir::Operation*> declareToAlloca;

        /// This is the last alloca that was created (either for PRINT/ERROR or for a user defined variable.)
        ///
        /// These are now hoisted to the top of the function scope, and will be created one after the other.
        mlir::Operation* lastAlloca{};
    };

    /// Type for mapping DebugNameOps operation pointers to DILexicalBlockAttr
    using DebugScopeMap = llvm::DenseMap<mlir::Operation*, mlir::LLVM::DILexicalBlockAttr>;

    /// ScopeBeginOp, ScopeEndOp for a given level and it's parent level.
    struct ScopeRecord
    {
        /// The ScopeBeginOp
        mlir::Operation* beginOp{};

        /// The ScopeEndOp
        mlir::Operation* endOp{};

        /// The DILexicalBlockAttr that will be created for the given pair of ScopeBeginOp,ScopeEndOp operations.
        mlir::LLVM::DILexicalBlockAttr lexicalBlock{};

        /// -1 means rooted at subprogram
        int32_t parentId{ -1 };
    };

    /// Context object holding state and helper functions used during lowering
    /// of the Silly dialect to LLVM dialect.
    class LoweringContext
    {
       public:
        /// Some common mlir::Type values.
        MlirTypeCache typ;

        /// LLVM type for PRINT builtin arguments (cached for convenience).
        mlir::Type printArgStructTy;

        /// Constructs the lowering context.
        /// @param moduleOp The module being lowered.
        /// @param ds Reference to driver configuration state.
        LoweringContext( mlir::ModuleOp& moduleOp, DriverState& ds );

        /// Returns the preferred alignment for an element type according to the data layout.
        unsigned preferredTypeAlignment( mlir::Operation* op, mlir::Type elemType );

        /// Looks up or creates a global constant for a string literal.
        mlir::LLVM::GlobalOp lookupOrInsertGlobalOp( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                     mlir::StringAttr& stringLit, size_t strLen );

        /// Creates a call to the appropriate Silly print runtime function.
        mlir::LogicalResult emitPrintArgStruct( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                mlir::Operation* op, mlir::Value input, PrintFlags flags,
                                                mlir::Value& output );

        /// A CallOp generator for the __silly_abort runtime function.
        void createAbortCall( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );

        /// Creates a call to the appropriate Silly get runtime function.
        mlir::LogicalResult createGetCall( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                           mlir::Operation* op, mlir::Type inputType, mlir::Value& output );

        /// Emits debug information for a local variable (scalar or array), or a parameter, or a FOR induction variable.
        mlir::LogicalResult constructVariableDI( mlir::ConversionPatternRewriter& rewriter, silly::DebugNameOp );

        /// Emits debug information for a function parameter.
        void constructParameterDI( mlir::FileLineColLoc loc, mlir::ConversionPatternRewriter& rewriter,
                                   const std::string& varName, mlir::LLVM::AllocaOp value, mlir::Type elemType,
                                   int paramIndex, const std::string& funcName );

        /// Return the PRINT args allocation for this function, big enough for the biggest PRINT list in the function.
        mlir::LLVM::AllocaOp getPrintArgs( const std::string& funcName );

        /// Retrieve the allocation associated with a DeclareOp
        mlir::LLVM::AllocaOp getAlloca( const std::string& funcName, mlir::Operation* dclOp );

        /// Cache the allocation associated with a DeclareOp
        void setAlloca( const std::string& funcName, mlir::Operation* dclOp, mlir::Operation* aOp );

        /// Helper function to generate MLIR for a silly language assignment statement (`foo = bar;`)
        mlir::LogicalResult generateAssignment( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                mlir::Operation*, mlir::Value value, mlir::Type elemType,
                                                mlir::LLVM::AllocaOp allocaOp, unsigned alignment,
                                                mlir::TypedValue<mlir::IndexType> optIndex );

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
        inline mlir::LogicalResult getIzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                             mlir::Operation* op, unsigned width, mlir::LLVM::ConstantOp& output );

        /// Creates an i64 one constant.
        inline mlir::LLVM::ConstantOp getI64one( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter );


        /// Creates the DICompileUnit and basic debug types if debugging is enabled.
        void createDICompileUnit( mlir::Location loc );

        /// Emits debug metadata for a function if debugging is enabled.
        /// @retval true if error
        bool createPerFuncState( mlir::func::FuncOp funcOp );

        /// Looks up an existing global for a string literal.
        mlir::LLVM::GlobalOp lookupGlobalOp( mlir::StringAttr& stringLit );

        /// Add a memset builtin with the user supplied (or default) fill value.
        void insertFill( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, mlir::LLVM::AllocaOp allocaOp,
                         mlir::Value bytesVal );

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

        /// mlir::LLVM::FRemOp lowers to fmod (at least on some targets), so -lm will be required at link time.
        void markMathLibRequired();

        /// Walk ops until hitting an ScopeBeginOp, and if found call processScopeBegin.
        void processScopedOps( mlir::func::FuncOp funcOp, mlir::LLVM::DIScopeAttr rootScope );

        /// @brief process all the ops until a matching ScopeEndOp is found.
        ///
        /// Create a DILexicalBlockAttr for the scope.
        /// Record the DILexicalBlockAttr that should apply to any DebugNameOps found.
        /// Restamp the OP locations with a fused location that ties the OP to the DILexicalBlockAttr.
        /// If an OP has any regions, process those recursively.
        mlir::Block::iterator processScopeBegin( mlir::Block::iterator it, mlir::Block::iterator blockEnd,
                                                 mlir::LLVM::DIScopeAttr parentScope );


        /// Walk the operations in a FuncOp and find all the ScopeBeginOp ScopeEndOp
        void collectScopeOps( mlir::func::FuncOp funcOp, llvm::DenseMap<int32_t, ScopeRecord>& scopeRecords );

        /// Returns the stack snapshot to propagate to successors.
        llvm::SmallVector<int32_t> processBlock( mlir::Block* block, llvm::SmallVector<int32_t> scopeStack,
                                                 llvm::DenseMap<int32_t, ScopeRecord>& scopeRecords,
                                                 mlir::LLVM::DIScopeAttr rootScope );

        /// Set the IP to the funcOp start position, or just after the last alloca, then create the AllocaOp
        /// and save that AllocaOp's Operation* to lastAlloca.
        mlir::LLVM::AllocaOp createAlloca( mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op,
                                           mlir::Type elemType, int64_t arraySize, unsigned alignment );

       private:
        /// Returns the MLIR context.
        inline mlir::MLIRContext* getContext();

        /// Creates the prototype for __silly_print.
        inline void createSillyPrintPrototype();

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

        /// Debug file attribute (used when debugging is enabled).
        mlir::LLVM::DIFileAttr fileAttr;

        /// Map from function name to its DISubprogram attribute (and other stuff)
        std::unordered_map<std::string, PerFunctionLoweringState> lookupFunctionState;

        /// Type for mapping from string literal content to its GlobalOp.
        using StringLit2GlobalOp = std::unordered_map<std::string, mlir::LLVM::GlobalOp>;

        /// Map from string literal content to its GlobalOp.
        StringLit2GlobalOp stringLiterals;

        /// Map debugNameOp to DILexicalBlock
        DebugScopeMap scopeMap;

        /// Map from the first op after a scope_begin to the scope's closing location
        llvm::DenseMap<mlir::Block*, std::optional<mlir::Location>> blockClosingLoc;

        /// Prototype for __silly_print.
        mlir::func::FuncOp printFunc;

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
        DriverState& driverState;

        /// The module being lowered.
        mlir::ModuleOp& mod;

        /// OpBuilder positioned in the module.
        mlir::OpBuilder builder;

        /// Type converter for the lowering pass.
        mlir::LLVMTypeConverter typeConverter;

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

    inline mlir::LLVM::ConstantOp LoweringContext::getI64one( mlir::Location loc,
                                                              mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.i64, rewriter.getI64IntegerAttr( 1 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getF32zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.f32, rewriter.getF32FloatAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getF64zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.f64, rewriter.getF64FloatAttr( 0 ) );
    }

    inline mlir::MLIRContext* LoweringContext::getContext()
    {
        return builder.getContext();
    }

    inline bool LoweringContext::isTypeFloat( mlir::Type ty ) const
    {
        return ( ( ty == typ.f32 ) || ( ty == typ.f64 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI8zero( mlir::Location loc,
                                                              mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.i8, rewriter.getI8IntegerAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI16zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.i16, rewriter.getI16IntegerAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI32zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.i32, rewriter.getI32IntegerAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI64zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return mlir::LLVM::ConstantOp::create( rewriter, loc, typ.i64, rewriter.getI64IntegerAttr( 0 ) );
    }

    /// Returns a zero constant for the given integer width (i8, i16, i32, i64).
    inline mlir::LogicalResult LoweringContext::getIzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                          mlir::Operation* op, unsigned width,
                                                          mlir::LLVM::ConstantOp& output )
    {
        switch ( width )
        {
            case 8:
            {
                output = getI8zero( loc, rewriter );
                break;
            }
            case 16:
            {
                output = getI16zero( loc, rewriter );
                break;
            }
            case 32:
            {
                output = getI32zero( loc, rewriter );
                break;
            }
            case 64:
            {
                output = getI64zero( loc, rewriter );
                break;
            }
            default:
            {
                return rewriter.notifyMatchFailure( op, llvm::formatv( "Unexpected integer size: {0}", width ) );
            }
        }

        return mlir::success();
    }

    inline mlir::LLVMTypeConverter& LoweringContext::getTypeConverter()
    {
        return typeConverter;
    }
}    // namespace silly

// vim: et ts=4 sw=4
