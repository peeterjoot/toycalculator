///
/// @file    lowering.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This file implements the LLVM-IR lowering pattern matching operators
///
#include <llvm/ADT/StringRef.h>
#include <llvm/BinaryFormat/Dwarf.h>    // For DW_LANG_C, DW_ATE_*
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <format>
#include <numeric>

#include "lowering.hpp"
#include "loweringContext.hpp"

/// --debug- type for lowering
#define DEBUG_TYPE "silly-lowering"

/// For llvm.ident
#define COMPILER_VERSION " V8"

namespace silly
{
    silly::ScopeOp getEnclosingScopeOp( mlir::Location loc, mlir::func::FuncOp funcOp );

    /// Set and restore an insertion point using a RAII model.
    class ModuleInsertionPointGuard
    {
       public:
        /// Constructor, that sets the insertion point to the beginning of the module.
        ///
        /// Also saves the current insertion point.
        ModuleInsertionPointGuard( mlir::ModuleOp& mod, mlir::OpBuilder& opBuilder )
            : builder{ opBuilder }, oldIP{ builder.saveInsertionPoint() }
        {
            builder.setInsertionPointToStart( mod.getBody() );
        }

        /// Destructor, that restores the original insertion point.
        ~ModuleInsertionPointGuard()
        {
            builder.restoreInsertionPoint( oldIP );
        }

       private:
        mlir::OpBuilder& builder;    ///< cache the builder for IP restoration.

        mlir::OpBuilder::InsertPoint oldIP;    ///< the old IP
    };

    /// Assuming that a Location is actually a FileLineColLoc, cast it and return it as so.
    ///
    /// Will assert if this is not the case.
    mlir::FileLineColLoc getLocation( mlir::Location loc )
    {
        // Cast Location to FileLineColLoc
        mlir::FileLineColLoc fileLineLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        assert( fileLineLoc );

        return fileLineLoc;
    }

    /// Find the mlir::func::FuncOp that contains the provided op.
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

    LoweringContext::LoweringContext( mlir::ModuleOp& moduleOp, silly::DriverState& ds )
        : driverState{ ds }, mod{ moduleOp }, builder{ mod.getRegion() }, typeConverter{ builder.getContext() }
    {
        // Configure the type converter to handle silly::VarType -> !llvm.ptr
        typeConverter.addConversion( []( silly::varType type ) -> mlir::Type
                                     { return mlir::LLVM::LLVMPointerType::get( type.getContext() ); } );

        tyI1 = builder.getI1Type();
        tyI8 = builder.getI8Type();
        tyI16 = builder.getI16Type();
        tyI32 = builder.getI32Type();
        tyI64 = builder.getI64Type();

        tyF32 = builder.getF32Type();
        tyF64 = builder.getF64Type();

        mlir::MLIRContext* context = builder.getContext();
        tyPtr = mlir::LLVM::LLVMPointerType::get( context );

        tyVoid = mlir::LLVM::LLVMVoidType::get( context );

        printArgStructTy =
            mlir::LLVM::LLVMStructType::getLiteral( context,
                                                    {
                                                        tyI32,    // kind: PrintKind (i32)
                                                        tyI32,    // flags: PrintFlags (i32)
                                                        tyI64,    // i, or string length, or bitcast double
                                                        tyPtr     // ptr: const char* (only used for STRING)
                                                    },
                                                    /*isPacked=*/false );
    }

    inline void LoweringContext::markMathLibRequired()
    {
        driverState.needsMathLib = true;
    }

    inline mlir::LLVMTypeConverter& LoweringContext::getTypeConverter()
    {
        return typeConverter;
    };

    unsigned LoweringContext::preferredTypeAlignment( mlir::Operation* op, mlir::Type elemType )
    {
        mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
        assert( mod );
        mlir::DataLayout dataLayout( mod );
        unsigned alignment = dataLayout.getTypePreferredAlignment( elemType );

        return alignment;
    }

    inline mlir::MLIRContext* LoweringContext::getContext()
    {
        return builder.getContext();
    }

    inline bool LoweringContext::isTypeFloat( mlir::Type ty ) const
    {
        return ( ( ty == tyF32 ) || ( ty == tyF64 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI8zero( mlir::Location loc,
                                                              mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI8, rewriter.getI8IntegerAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI16zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI16, rewriter.getI16IntegerAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI32zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI32, rewriter.getI32IntegerAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getI64zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( 0 ) );
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

    inline mlir::LLVM::ConstantOp LoweringContext::getI64one( mlir::Location loc,
                                                              mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( 1 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getF32zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyF32, rewriter.getF32FloatAttr( 0 ) );
    }

    inline mlir::LLVM::ConstantOp LoweringContext::getF64zero( mlir::Location loc,
                                                               mlir::ConversionPatternRewriter& rewriter )
    {
        return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyF64, rewriter.getF64FloatAttr( 0 ) );
    }

    void LoweringContext::createSillyPrintPrototype()
    {
        if ( !printFunc )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), { tyI32, tyPtr }, {} );

            printFunc = builder.create<mlir::func::FuncOp>( mod.getLoc(), "__silly_print", funcType );
            printFunc.setVisibility( mlir::SymbolTable::Visibility::Private );
        }
    }

    void LoweringContext::createSillyAbortPrototype()
    {
        if ( !printFuncAbort )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), { tyI64, tyPtr, tyI32 }, {} );
            printFuncAbort = builder.create<mlir::func::FuncOp>( mod.getLoc(), "__silly_abort", funcType );
            printFuncAbort.setVisibility( mlir::SymbolTable::Visibility::Private );
        }
    }

    template <class RetTy>
    void LoweringContext::createSillyGetPrototype( mlir::func::FuncOp& getOp, RetTy retType, const char* name )
    {
        if ( !getOp )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), {},    // no arguments
                                                                   { retType }                  // single return type
            );

            getOp = builder.create<mlir::func::FuncOp>( mod.getLoc(), name, funcType );
            getOp.setVisibility( mlir::SymbolTable::Visibility::Private );
        }
    }

    inline void LoweringContext::createSillyGetI1Prototype()
    {
        // returns int8_t, but checks input to verify 0/1 value.
        createSillyGetPrototype( getFuncI1, tyI8, "__silly_get_i1" );
    }

    inline void LoweringContext::createSillyGetI8Prototype()
    {
        createSillyGetPrototype( getFuncI8, tyI8, "__silly_get_i8" );
    }

    inline void LoweringContext::createSillyGetI16Prototype()
    {
        createSillyGetPrototype( getFuncI16, tyI16, "__silly_get_i16" );
    }

    inline void LoweringContext::createSillyGetI32Prototype()
    {
        createSillyGetPrototype( getFuncI32, tyI32, "__silly_get_i32" );
    }

    inline void LoweringContext::createSillyGetI64Prototype()
    {
        createSillyGetPrototype( getFuncI64, tyI64, "__silly_get_i64" );
    }

    inline void LoweringContext::createSillyGetF32Prototype()
    {
        createSillyGetPrototype( getFuncF32, tyF32, "__silly_get_f32" );
    }

    inline void LoweringContext::createSillyGetF64Prototype()
    {
        createSillyGetPrototype( getFuncF64, tyF64, "__silly_get_f64" );
    }

    void LoweringContext::createDICompileUnit()
    {
        if ( driverState.wantDebug )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::MLIRContext* context = builder.getContext();


            diVOID = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                       builder.getStringAttr( "void" ), 0, 0 );

            diF32 = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                      builder.getStringAttr( "float" ), 32, llvm::dwarf::DW_ATE_float );

            diF64 =
                mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                  builder.getStringAttr( "double" ), 64, llvm::dwarf::DW_ATE_float );

            diUNKNOWN = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                          builder.getStringAttr( "unknown" ), 0, 0 );


            diI8 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                     builder.getStringAttr( "char" ), 8,
                                                     (unsigned)llvm::dwarf::DW_ATE_signed );

            diI16 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                      builder.getStringAttr( "short" ), 16,
                                                      (unsigned)llvm::dwarf::DW_ATE_signed );

            diI32 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                      builder.getStringAttr( "int" ), 32,
                                                      (unsigned)llvm::dwarf::DW_ATE_signed );

            diI64 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                      builder.getStringAttr( "long" ), 64,
                                                      (unsigned)llvm::dwarf::DW_ATE_signed );

            // Construct module level DI state:
            fileAttr = mlir::LLVM::DIFileAttr::get( context, driverState.filename, "." );
            mlir::DistinctAttr distinctAttr = mlir::DistinctAttr::create( builder.getUnitAttr() );
            compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
                context, distinctAttr, llvm::dwarf::DW_LANG_C, fileAttr, builder.getStringAttr( COMPILER_NAME ), false,
                mlir::LLVM::DIEmissionKind::Full, mlir::LLVM::DINameTableKind::Default );
        }

        mod->setAttr( "llvm.ident", builder.getStringAttr( COMPILER_NAME COMPILER_VERSION ) );
    }

    mlir::LLVM::DITypeAttr LoweringContext::getDIType( mlir::Type type )
    {
        if ( !type )
        {
            return diVOID;
        }
        else if ( type.isF32() )
        {
            return diF32;
        }
        else if ( type.isF64() )
        {
            return diF64;
        }
        else if ( type.isInteger( 8 ) || type.isInteger( 1 ) )
        {
            return diI8;
        }
        else if ( type.isInteger( 16 ) )
        {
            return diI16;
        }
        else if ( type.isInteger( 32 ) )
        {
            return diI32;
        }
        else if ( type.isInteger( 64 ) )
        {
            return diI64;
        }
        else
        {
            return diUNKNOWN;
        }
    }

    mlir::LLVM::DISubroutineTypeAttr LoweringContext::createDISubroutineType( mlir::func::FuncOp funcOp )
    {
        mlir::FunctionType funcType = funcOp.getFunctionType();

        mlir::SmallVector<mlir::LLVM::DITypeAttr> paramTypes;

        mlir::LLVM::DITypeAttr returnType =
            getDIType( funcType.getResults().empty() ? mlir::Type() : funcType.getResults()[0] );
        paramTypes.push_back( returnType );

        for ( mlir::Type argType : funcType.getInputs() )
        {
            paramTypes.push_back( getDIType( argType ) );
        }

        mlir::MLIRContext* context = builder.getContext();

        return mlir::LLVM::DISubroutineTypeAttr::get( context, llvm::DINode::FlagZero, paramTypes );
    }

    bool LoweringContext::createPerFuncState( mlir::func::FuncOp funcOp )
    {
        size_t maxPrintArgs = 0;
        std::string funcName;

        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::MLIRContext* context = builder.getContext();
            funcName = funcOp.getSymName().str();

            mlir::Region& region = funcOp.getRegion();
            mlir::Block& entryBlock = region.front();

            if ( driverState.wantDebug )
            {
                mlir::LLVM::DISubroutineTypeAttr subprogramType = createDISubroutineType( funcOp );

                mlir::Location funcLoc = funcOp.getLoc();
                mlir::FileLineColLoc loc = getLocation( funcLoc );
                unsigned line = loc.getLine();
                unsigned scopeLine = line;

                // Get the location of the First operation in the block for the scopeLine:
                if ( !entryBlock.empty() )
                {
                    mlir::Operation* firstOp = &entryBlock.front();
                    mlir::Location firstLoc = firstOp->getLoc();
                    mlir::FileLineColLoc scopeLoc = getLocation( firstLoc );

                    scopeLine = scopeLoc.getLine();
                }

                mlir::LLVM::DISubprogramAttr sub = mlir::LLVM::DISubprogramAttr::get(
                    context, mlir::DistinctAttr::create( builder.getUnitAttr() ), compileUnitAttr, fileAttr,
                    builder.getStringAttr( funcName ), builder.getStringAttr( funcName ), fileAttr, line, scopeLine,
                    mlir::LLVM::DISubprogramFlags::Definition, subprogramType, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{},
                    llvm::ArrayRef<mlir::LLVM::DINodeAttr>{} );

                funcOp->setAttr( "llvm.debug.subprogram", sub );

                // This is the key to ensure that translateModuleToLLVMIR does not strip the location info (instead
                // converts loc's into !dbg's)
                funcOp->setLoc( builder.getFusedLoc( { mod.getLoc() }, sub ) );

                funcState[funcName].subProgramDI = sub;
            }

            // Compute max print args for this function
            funcOp.walk(
                [&]( silly::PrintOp printOp )
                {
                    auto ins = printOp.getInputs();
                    size_t n = ins.size();

                    if ( n > maxPrintArgs )
                    {
                        maxPrintArgs = n;
                    }
                } );
        }

        {
            mlir::OpBuilder::InsertPoint savedIP = builder.saveInsertionPoint();

            mlir::Location loc = builder.getUnknownLoc();

            silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );
            if ( !scopeOp )
            {
                return true;
            }

            mlir::Block* scopeBlock = &scopeOp.getBody().front();

            builder.setInsertionPointToStart( scopeBlock );

            funcState[funcName].printArgs = builder.create<mlir::LLVM::AllocaOp>(
                loc, tyPtr, printArgStructTy,
                builder.create<mlir::LLVM::ConstantOp>( loc, tyI64, builder.getI64IntegerAttr( maxPrintArgs ) ) );

            builder.restoreInsertionPoint( savedIP );
        }

        return false;
    }

    mlir::LLVM::AllocaOp LoweringContext::getPrintArgs( const std::string& funcName )
    {
        return funcState[funcName].printArgs;
    }

    mlir::LLVM::AllocaOp LoweringContext::getAlloca( const std::string& funcName, mlir::Operation* dclOp )
    {
        mlir::Operation* aOp = funcState[funcName].declareToAlloca[dclOp];
        mlir::LLVM::AllocaOp allocOp = mlir::cast<mlir::LLVM::AllocaOp>( aOp );

        return allocOp;
    }

    void LoweringContext::setAlloca( const std::string& funcName, mlir::Operation* dclOp, mlir::Operation* aOp )
    {
        funcState[funcName].declareToAlloca[dclOp] = aOp;
    }

    /// Looks up the enclosing function name for an operation.
    static std::string lookupFuncNameForOp( mlir::Operation* op )
    {
        mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

        return funcOp.getSymName().str();
    }

    mlir::LogicalResult LoweringContext::infoForVariableDI( mlir::FileLineColLoc loc,
                                                            mlir::ConversionPatternRewriter& rewriter,
                                                            mlir::Operation* op, llvm::StringRef varName,
                                                            mlir::Type& elemType, unsigned elemSizeInBits,
                                                            int64_t arraySize, const char*& typeName, unsigned& dwType,
                                                            unsigned& elemStorageSizeInBits )
    {
        if ( !driverState.wantDebug )
        {
            return mlir::success();
        }

        typeName = nullptr;
        dwType = llvm::dwarf::DW_ATE_signed;
        elemStorageSizeInBits = elemSizeInBits;    // Storage size (e.g., i1 uses i8)

        if ( mlir::isa<mlir::IntegerType>( elemType ) )
        {
            switch ( elemSizeInBits )
            {
                case 1:
                {
                    typeName = "bool";
                    dwType = llvm::dwarf::DW_ATE_boolean;
                    elemStorageSizeInBits = 8;
                    break;
                }
                case 8:
                {
                    typeName = "char";    // Using "char" for STRING arrays
                    dwType = llvm::dwarf::DW_ATE_signed_char;
                    break;
                }
                case 16:
                {
                    typeName = "int16_t";
                    break;
                }
                case 32:
                {
                    typeName = "int32_t";
                    break;
                }
                case 64:
                {
                    typeName = "int64_t";
                    break;
                }
                default:
                {
                    return rewriter.notifyMatchFailure(
                        op, llvm::formatv( "Unsupported integer type size: {0}", elemSizeInBits ) );
                }
            }
        }
        else if ( mlir::isa<mlir::FloatType>( elemType ) )
        {
            dwType = llvm::dwarf::DW_ATE_float;
            switch ( elemSizeInBits )
            {
                case 32:
                {
                    typeName = "float";
                    break;
                }
                case 64:
                {
                    typeName = "double";
                    break;
                }
                default:
                {
                    return rewriter.notifyMatchFailure(
                        op, llvm::formatv( "Unsupported float type size: {0}", elemSizeInBits ) );
                }
            }
        }
        else
        {
            return rewriter.notifyMatchFailure( op, llvm::formatv( "Unsupported type for debug info {0}", elemType ) );
        }

        return mlir::success();
    }

    mlir::LogicalResult LoweringContext::constructInductionVariableDI( mlir::FileLineColLoc fileLoc,
                                                                       mlir::ConversionPatternRewriter& rewriter,
                                                                       mlir::Operation* op, mlir::Value value,
                                                                       std::string varName, mlir::StringAttr nameAttr,
                                                                       mlir::Type elemType, unsigned elemSizeInBits,
                                                                       std::string funcName )
    {
        assert( fileLoc );

        mlir::LLVM::DISubprogramAttr sub = funcState[funcName].subProgramDI;
        assert( sub );

        const char* typeName;
        unsigned dwType;
        unsigned elemStorageSizeInBits;

        if ( mlir::failed( LoweringContext::infoForVariableDI( fileLoc, rewriter, op, varName, elemType, elemSizeInBits,
                                                               1, typeName, dwType, elemStorageSizeInBits ) ) )
        {
            return mlir::failure();
        }

        mlir::MLIRContext* context = builder.getContext();

        mlir::LLVM::DITypeAttr diType = mlir::LLVM::DIBasicTypeAttr::get(
            context, llvm::dwarf::DW_TAG_base_type, builder.getStringAttr( typeName ), elemStorageSizeInBits, dwType );

        mlir::LLVM::DILocalVariableAttr diVar = mlir::LLVM::DILocalVariableAttr::get(
            context, sub, nameAttr, fileAttr, fileLoc.getLine(),
            /*argNo=*/0, elemStorageSizeInBits, diType, mlir::LLVM::DIFlags::Zero );

        // Emit llvm.dbg.value
        // Empty expression for direct value binding
        mlir::LLVM::DIExpressionAttr emptyExpr = mlir::LLVM::DIExpressionAttr::get( context, {} );

        rewriter.create<mlir::LLVM::DbgValueOp>( fileLoc, value, diVar, emptyExpr );

        return mlir::success();
    }

    mlir::LogicalResult LoweringContext::constructVariableDI( mlir::FileLineColLoc loc,
                                                              mlir::ConversionPatternRewriter& rewriter,
                                                              mlir::Operation* op, llvm::StringRef varName,
                                                              mlir::Type& elemType, unsigned elemSizeInBits,
                                                              mlir::LLVM::AllocaOp& allocaOp, int64_t arraySize )
    {
        if ( driverState.wantDebug )
        {
            mlir::MLIRContext* context = builder.getContext();

            allocaOp->setAttr( "bindc_name", builder.getStringAttr( varName ) );

            mlir::LLVM::DILocalVariableAttr diVar;
            mlir::LLVM::DITypeAttr diType;

            const char* typeName;
            unsigned dwType;
            unsigned elemStorageSizeInBits;

            if ( mlir::failed( LoweringContext::infoForVariableDI( loc, rewriter, op, varName, elemType, elemSizeInBits,
                                                                   arraySize, typeName, dwType,
                                                                   elemStorageSizeInBits ) ) )
            {
                return mlir::failure();
            }

            std::string funcName = lookupFuncNameForOp( allocaOp );
            mlir::LLVM::DISubprogramAttr sub = funcState[funcName].subProgramDI;
            assert( sub );

            unsigned totalSizeInBits = elemStorageSizeInBits * arraySize;
            if ( arraySize > 1 )
            {
                // Create base type for array elements
                mlir::LLVM::DIBasicTypeAttr baseType = mlir::LLVM::DIBasicTypeAttr::get(
                    context, llvm::dwarf::DW_TAG_base_type, builder.getStringAttr( typeName ), elemStorageSizeInBits,
                    dwType );

                // Create subrange for array (count = arraySize, lowerBound = 0)
                mlir::IntegerAttr countAttr = mlir::IntegerAttr::get( tyI64, arraySize );
                mlir::IntegerAttr lowerBoundAttr = mlir::IntegerAttr::get( tyI64, 0 );
                mlir::LLVM::DISubrangeAttr subrange =
                    mlir::LLVM::DISubrangeAttr::get( context, countAttr, lowerBoundAttr,
                                                     /*upperBound=*/nullptr, /*stride=*/nullptr );

                // Create array type
                unsigned alignInBits = elemStorageSizeInBits;    // Alignment matches element size
                diType = mlir::LLVM::DICompositeTypeAttr::get(
                    context, llvm::dwarf::DW_TAG_array_type, builder.getStringAttr( "" ), fileAttr,
                    /*line=*/0, sub, baseType, mlir::LLVM::DIFlags::Zero, totalSizeInBits, alignInBits,
                    llvm::ArrayRef<mlir::LLVM::DINodeAttr>{ subrange },
                    /*dataLocation=*/nullptr, /*rank=*/nullptr, /*allocated=*/nullptr, /*associated=*/nullptr );
            }
            else
            {
                // Scalar type
                diType = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                           builder.getStringAttr( typeName ), elemStorageSizeInBits,
                                                           dwType );
            }

            diVar = mlir::LLVM::DILocalVariableAttr::get(
                context, sub, builder.getStringAttr( varName ), fileAttr, loc.getLine(),
                /*argNo=*/0, totalSizeInBits, diType, mlir::LLVM::DIFlags::Zero );

            builder.setInsertionPointAfter( allocaOp );
            builder.create<mlir::LLVM::DbgDeclareOp>( loc, allocaOp, diVar );
        }

        return mlir::success();
    }

    mlir::LLVM::GlobalOp LoweringContext::lookupGlobalOp( mlir::StringAttr& stringLit )
    {
        mlir::LLVM::GlobalOp globalOp;
        StringLit2GlobalOp::iterator it = stringLiterals.find( stringLit.str() );
        if ( it != stringLiterals.end() )
        {
            globalOp = it->second;
            LLVM_DEBUG( llvm::dbgs() << std::format( "Found global: {} for string literal '{}'\n",
                                                     globalOp.getSymName().str(), stringLit.str() ) );
        }

        return globalOp;
    }

    mlir::LLVM::GlobalOp LoweringContext::lookupOrInsertGlobalOp( mlir::Location loc,
                                                                  mlir::ConversionPatternRewriter& rewriter,
                                                                  mlir::StringAttr& stringLit, size_t strLen )
    {
        mlir::LLVM::GlobalOp globalOp;
        StringLit2GlobalOp::iterator it = stringLiterals.find( stringLit.str() );
        if ( it != stringLiterals.end() )
        {
            globalOp = it->second;
            LLVM_DEBUG( llvm::dbgs() << "Reusing global: " << globalOp.getSymName() << '\n' );
        }
        else
        {
            mlir::OpBuilder::InsertPoint savedIP = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointToStart( mod.getBody() );

            mlir::LLVM::LLVMArrayType arrayType = mlir::LLVM::LLVMArrayType::get( tyI8, strLen );

            mlir::SmallVector<char> stringData( stringLit.begin(), stringLit.end() );
            mlir::DenseElementsAttr denseAttr =
                mlir::DenseElementsAttr::get( mlir::RankedTensorType::get( { static_cast<int64_t>( strLen ) }, tyI8 ),
                                              mlir::ArrayRef<char>( stringData ) );

            std::string globalName = "str_" + std::to_string( stringLiterals.size() );
            globalOp = rewriter.create<mlir::LLVM::GlobalOp>( loc, arrayType, true, mlir::LLVM::Linkage::Private,
                                                              globalName, denseAttr );
            globalOp->setAttr( "unnamed_addr", rewriter.getUnitAttr() );

            stringLiterals[stringLit.str()] = globalOp;
            LLVM_DEBUG( llvm::dbgs() << "Created global: " << globalName << '\n' );

            rewriter.restoreInsertionPoint( savedIP );
        }

        return globalOp;
    }

    void LoweringContext::createAbortCall( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
    {
        mlir::FileLineColLoc fileLoc = getLocation( loc );

        const std::string& filename = fileLoc.getFilename().str();
        mlir::StringAttr strAttr = builder.getStringAttr( filename );

        mlir::LLVM::ConstantOp sizeConst =
            rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( filename.size() ) );

        std::string strValue = strAttr.getValue().str();
        size_t strLen = strValue.size();

        mlir::LLVM::GlobalOp globalOp = lookupOrInsertGlobalOp( loc, rewriter, strAttr, strLen );
        mlir::Value input = rewriter.create<mlir::LLVM::AddressOfOp>( loc, globalOp );

        mlir::LLVM::ConstantOp lineConst =
            rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI32, rewriter.getI32IntegerAttr( fileLoc.getLine() ) );

        createSillyAbortPrototype();
        const char* name = "__silly_abort";
        rewriter.create<silly::CallOp>( loc, mlir::TypeRange{}, name, mlir::ValueRange{ sizeConst, input, lineConst } );
    }

    // Returns the filled PrintArg struct value for one argument
    mlir::LogicalResult LoweringContext::emitPrintArgStruct( mlir::Location loc,
                                                             mlir::ConversionPatternRewriter& rewriter,
                                                             mlir::Operation* op, mlir::Value input, PrintFlags flags,
                                                             mlir::Value& output )
    {
        createSillyPrintPrototype();

        mlir::Type inputType = input.getType();
        mlir::Value structVal = rewriter.create<mlir::LLVM::UndefOp>( loc, printArgStructTy );

        PrintKind kind = PrintKind::UNKNOWN;
        mlir::Value valuePayload;    // for i or d or length
        mlir::Value strPtr = nullptr;

        if ( mlir::IntegerType intTy = mlir::dyn_cast<mlir::IntegerType>( inputType ) )
        {
            kind = PrintKind::I64;
            if ( intTy.getWidth() == 1 )
            {
                valuePayload = rewriter.create<mlir::LLVM::ZExtOp>( loc, tyI64, input );
            }
            else if ( intTy.getWidth() < 64 )
            {
                valuePayload = rewriter.create<mlir::LLVM::SExtOp>( loc, tyI64, input );
            }
            else
            {
                valuePayload = input;
            }
        }
        else if ( mlir::FloatType floatTy = mlir::dyn_cast<mlir::FloatType>( inputType ) )
        {
            kind = PrintKind::F64;
            if ( inputType == tyF32 )
            {
                valuePayload = rewriter.create<mlir::LLVM::FPExtOp>( loc, tyF64, input );
            }
            else
            {
                valuePayload = input;
            }

            valuePayload = rewriter.create<mlir::LLVM::BitcastOp>( loc, tyI64, valuePayload );
        }
        else if ( inputType == tyPtr )
        {
            kind = PrintKind::STRING;

            int64_t numElems = 0;
            mlir::Value ptr = input;

            if ( silly::LoadOp loadOp = input.getDefiningOp<silly::LoadOp>() )
            {
                mlir::Value var = loadOp.getVar();
                assert( var );
                mlir::LLVM::AllocaOp allocaOp = var.getDefiningOp<mlir::LLVM::AllocaOp>();
                assert( allocaOp );
                if ( allocaOp.getElemType() != tyI8 )
                {
                    return rewriter.notifyMatchFailure( op, "expected i8 alloca type." );
                }

                if ( mlir::LLVM::ConstantOp constOp = allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
                {
                    numElems = mlir::cast<mlir::IntegerAttr>( constOp.getValue() ).getInt();
                }
            }
            else if ( silly::StringLiteralOp stringLitOp = input.getDefiningOp<silly::StringLiteralOp>() )
            {
                mlir::StringAttr strAttr = stringLitOp.getValueAttr();
                llvm::StringRef strValue = strAttr.getValue();
                numElems = strValue.size();
                mlir::LLVM::GlobalOp globalOp = lookupGlobalOp( strAttr );
                ptr = rewriter.create<mlir::LLVM::AddressOfOp>( loc, globalOp );
            }
            else
            {
                return rewriter.notifyMatchFailure( op, "unsupported string source" );
            }

            valuePayload =
                rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( numElems ) );
            strPtr = ptr;
        }
        else
        {
            return rewriter.notifyMatchFailure( op, "unsupported print argument type" );
        }

        // Insert kind (index 0)
        mlir::LLVM::ConstantOp kindVal = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, tyI32, rewriter.getI32IntegerAttr( static_cast<uint32_t>( kind ) ) );
        structVal = rewriter.create<mlir::LLVM::InsertValueOp>( loc, printArgStructTy, structVal, kindVal, 0 );

        // Insert flags (index 1)
        mlir::LLVM::ConstantOp flagsVal = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, tyI32, rewriter.getI32IntegerAttr( static_cast<uint32_t>( flags ) ) );
        structVal = rewriter.create<mlir::LLVM::InsertValueOp>( loc, printArgStructTy, structVal, flagsVal, 1 );

        // Insert "union" payload (index 2: i or d or length)
        structVal = rewriter.create<mlir::LLVM::InsertValueOp>( loc, printArgStructTy, structVal, valuePayload, 2 );

        // Insert string pointer if needed (index 3)
        if ( kind == PrintKind::STRING )
        {
            structVal = rewriter.create<mlir::LLVM::InsertValueOp>( loc, printArgStructTy, structVal, strPtr, 3 );
        }
        else
        {
            mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>( loc, tyPtr );

            structVal = rewriter.create<mlir::LLVM::InsertValueOp>( loc, printArgStructTy, structVal, nullPtr, 3 );
        }

        output = structVal;
        return mlir::success();
    }

    mlir::LogicalResult LoweringContext::createGetCall( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                        mlir::Operation* op, mlir::Type inputType, mlir::Value& output )
    {
        const char* name = nullptr;
        bool isBool{};

        if ( mlir::IntegerType inputi = mlir::dyn_cast<mlir::IntegerType>( inputType ) )
        {
            unsigned width = inputi.getWidth();

            switch ( width )
            {
                case 1:
                {
                    name = "__silly_get_i1";
                    createSillyGetI1Prototype();
                    inputType = tyI8;
                    isBool = true;
                    break;
                }
                case 8:
                {
                    name = "__silly_get_i8";
                    createSillyGetI8Prototype();
                    break;
                }
                case 16:
                {
                    name = "__silly_get_i16";
                    createSillyGetI16Prototype();
                    break;
                }
                case 32:
                {
                    name = "__silly_get_i32";
                    createSillyGetI32Prototype();
                    break;
                }
                case 64:
                {
                    name = "__silly_get_i64";
                    createSillyGetI64Prototype();
                    break;
                }
                default:
                {
                    return rewriter.notifyMatchFailure( op, "Unexpected integer size." );
                }
            }
        }
        else if ( mlir::FloatType inputf = mlir::dyn_cast<mlir::FloatType>( inputType ) )
        {
            if ( inputType == tyF32 )
            {
                name = "__silly_get_f32";
                createSillyGetF32Prototype();
            }
            else if ( inputType == tyF64 )
            {
                name = "__silly_get_f64";
                createSillyGetF64Prototype();
            }
            else
            {
                return rewriter.notifyMatchFailure( op, "Unexpected floating point type." );
            }
        }
        else
        {
            return rewriter.notifyMatchFailure( op, "Unsupported type." );
        }

        silly::CallOp callOp =
            rewriter.create<silly::CallOp>( loc, mlir::TypeRange{ inputType }, name, mlir::ValueRange{} );
        mlir::Value result = *callOp.getResult().begin();

        if ( isBool )
        {
            result = rewriter.create<mlir::LLVM::TruncOp>( loc, tyI1, result );
        }

        output = result;
        return mlir::success();
    }

    /// Emit debug info for parameter
    void LoweringContext::constructParameterDI( mlir::FileLineColLoc loc, mlir::ConversionPatternRewriter& rewriter,
                                                const std::string& varName, mlir::LLVM::AllocaOp value,
                                                mlir::Type elemType, int paramIndex, const std::string& funcName )
    {
        if ( driverState.wantDebug )
        {
            // Create debug type for basic types (e.g., i32, f32)
            mlir::MLIRContext* context = rewriter.getContext();
            mlir::LLVM::DITypeAttr diType = getDIType( elemType );

            mlir::LLVM::DISubprogramAttr sub = funcState[funcName].subProgramDI;
            assert( sub );

            unsigned bitWidth = elemType.getIntOrFloatBitWidth();

            // Create debug variable
            mlir::LLVM::DILocalVariableAttr diVar = mlir::LLVM::DILocalVariableAttr::get(
                context, sub, rewriter.getStringAttr( varName ), fileAttr, loc.getLine(), paramIndex + 1, bitWidth,
                diType, mlir::LLVM::DIFlags::Zero );

            // Emit llvm.dbg.declare
            rewriter.create<mlir::LLVM::DbgDeclareOp>( loc, value, diVar );
        }
    }

    mlir::Value LoweringContext::castToElemType( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                 mlir::Value value, mlir::Type valType, mlir::Type elemType )
    {
        if ( valType == tyF64 )
        {
            if ( mlir::isa<mlir::IntegerType>( elemType ) )
            {
                value = rewriter.create<mlir::LLVM::FPToSIOp>( loc, elemType, value );
            }
            else if ( elemType == tyF32 )
            {
                value = rewriter.create<mlir::LLVM::FPTruncOp>( loc, elemType, value );
            }
        }
        else if ( valType == tyF32 )
        {
            if ( mlir::isa<mlir::IntegerType>( elemType ) )
            {
                value = rewriter.create<mlir::LLVM::FPToSIOp>( loc, elemType, value );
            }
            else if ( elemType == tyF64 )
            {
                value = rewriter.create<mlir::LLVM::FPExtOp>( loc, elemType, value );
            }
        }
        else if ( mlir::IntegerType viType = mlir::cast<mlir::IntegerType>( valType ) )
        {
            unsigned vwidth = viType.getWidth();
            if ( isTypeFloat( elemType ) )
            {
                if ( vwidth == 1 )
                {
                    value = rewriter.create<mlir::LLVM::UIToFPOp>( loc, elemType, value );
                }
                else
                {
                    value = rewriter.create<mlir::LLVM::SIToFPOp>( loc, elemType, value );
                }
            }
            else if ( mlir::IntegerType miType = mlir::cast<mlir::IntegerType>( elemType ) )
            {
                unsigned mwidth = miType.getWidth();
                if ( vwidth > mwidth )
                {
                    value = rewriter.create<mlir::LLVM::TruncOp>( loc, elemType, value );
                }
                else if ( vwidth < mwidth )
                {
                    value = rewriter.create<mlir::LLVM::ZExtOp>( loc, elemType, value );
                }
            }
        }

        return value;
    }

    mlir::LogicalResult LoweringContext::generateAssignment( mlir::Location loc,
                                                             mlir::ConversionPatternRewriter& rewriter,
                                                             mlir::Operation* op, mlir::Value value,
                                                             mlir::Type elemType, mlir::LLVM::AllocaOp allocaOp,
                                                             unsigned alignment,
                                                             mlir::TypedValue<mlir::IndexType> optIndex )
    {
        mlir::Type valType = value.getType();

        // varName: i1v
        // value: %true = arith.constant true
        // valType: i1
        LLVM_DEBUG( llvm::dbgs() << "value: " << value << '\n' );
        LLVM_DEBUG( llvm::dbgs() << "valType: " << valType << '\n' );

        // extract parameters from the allocaOp so we know what to do here:
        int64_t numElems = 0;
        if ( mlir::LLVM::ConstantOp constOp = allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
        {
            mlir::IntegerAttr intAttr = mlir::dyn_cast<mlir::IntegerAttr>( constOp.getValue() );
            numElems = intAttr.getInt();
        }

        // LLVM_DEBUG( llvm::dbgs() << "memType: " << memType << '\n' );
        LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );
        // LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );

        if ( numElems == 1 )
        {
            value = castToElemType( loc, rewriter, value, valType, elemType );

            rewriter.create<mlir::LLVM::StoreOp>( loc, value, allocaOp, alignment );
        }
        else if ( silly::StringLiteralOp stringLitOp = value.getDefiningOp<silly::StringLiteralOp>() )
        {
            if ( elemType != tyI8 )
            {
                return rewriter.notifyMatchFailure(
                    op, llvm::formatv( "string assignment requires i8 array.  have: {0}", elemType ) );
            }
            if ( numElems == 0 )
            {
                return rewriter.notifyMatchFailure( op, "invalid zero size array" );
            }

            mlir::StringAttr strAttr = stringLitOp.getValueAttr();
            llvm::StringRef strValue = strAttr.getValue();
            size_t literalStrLen = strValue.size();
            mlir::LLVM::GlobalOp globalOp = lookupGlobalOp( strAttr );

            mlir::LLVM::AddressOfOp globalPtr = rewriter.create<mlir::LLVM::AddressOfOp>( loc, globalOp );

            mlir::Value destPtr = allocaOp.getResult();

            int copySize = std::min( (int)numElems, (int)literalStrLen );
            mlir::LLVM::ConstantOp sizeConst =
                rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( copySize ) );

            rewriter.create<mlir::LLVM::MemcpyOp>( loc, destPtr, globalPtr, sizeConst, rewriter.getBoolAttr( false ) );

            // If target array is larger than string literal, zero out the remaining bytes
            if ( numElems > (int64_t)literalStrLen )
            {
                // Compute the offset: destPtr + literalStrLen
                mlir::LLVM::ConstantOp offsetConst =
                    rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( literalStrLen ) );
                mlir::LLVM::GEPOp destPtrOffset = rewriter.create<mlir::LLVM::GEPOp>(
                    loc, destPtr.getType(), elemType, destPtr, mlir::ValueRange{ offsetConst } );

                // Compute the number of bytes to zero: numElems - literalStrLen
                mlir::LLVM::ConstantOp remainingSize = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, tyI64, rewriter.getI64IntegerAttr( numElems - literalStrLen ) );

                // Set remaining bytes to zero
                rewriter.create<mlir::LLVM::MemsetOp>( loc, destPtrOffset, getI8zero( loc, rewriter ), remainingSize,
                                                       rewriter.getBoolAttr( false ) );
            }
        }
        else    // ARRAY ELEMENT or UNSUPPORTED ASSIGNMENT
        {
            if ( !optIndex )
            {
                // Assigning a non-string-literal to an array (e.g., t = some_expr;)
                // This is not supported (arrays are not first-class values)
                return rewriter.notifyMatchFailure(
                    op, "assignment of non-string-literal to array variable without index is not supported" );
            }

            mlir::Value indexVal = optIndex;
            mlir::Value destBasePtr = allocaOp.getResult();

            if ( !numElems )
            {
                return rewriter.notifyMatchFailure(
                    op, "non-scalar, non-string assignment must be an array with non-zero size" );
            }

            if ( mlir::arith::ConstantIndexOp constOp = indexVal.getDefiningOp<mlir::arith::ConstantIndexOp>() )
            {
                int64_t idx = constOp.value();
                if ( idx < 0 || idx >= numElems )
                {
                    return rewriter.notifyMatchFailure(
                        op, llvm::formatv(
                                "static out-of-bounds array access: index {0} is out of bounds for array of size {1}",
                                idx, numElems ) );
                }
            }

            // Cast index to i64 for LLVM dialect GEP indexing
            mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>( loc, tyI64, indexVal );

            mlir::Type elemPtrTy = destBasePtr.getType();

            mlir::Value elemPtr = rewriter.create<mlir::LLVM::GEPOp>( loc,
                                                                      elemPtrTy,    // result type
                                                                      elemType,     // pointee type
                                                                      destBasePtr, mlir::ValueRange{ idxI64 } );

            // Nice to have (untested): Runtime bounds check -- make this a compile option?
            // if (numElems > 0) {
            //     mlir::Value sizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
            //         loc, tyI64, rewriter.getI64IntegerAttr(numElems));
            //     mlir::Value inBounds = rewriter.create<mlir::LLVM::ICmpOp>(
            //         loc, mlir::LLVM::ICmpPredicate::ult, idxI64, sizeVal);
            //
            //     mlir::Block* trapBB = rewriter.createBlock(rewriter.getInsertionBlock()->getParent());
            //     mlir::Block* contBB = rewriter.createBlock(trapBB);
            //
            //     rewriter.create<mlir::LLVM::CondBrOp>(loc, inBounds, contBB, trapBB);
            //
            //     rewriter.setInsertionPointToStart(trapBB);
            //     rewriter.create<mlir::LLVM::UnreachableOp>(loc);  // or call abort()
            //
            //     rewriter.setInsertionPointToStart(contBB);
            // }

            value = castToElemType( loc, rewriter, value, valType, elemType );

            rewriter.create<mlir::LLVM::StoreOp>( loc, value, elemPtr, alignment );
        }

        return mlir::success();
    }

    void LoweringContext::insertFill( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                      mlir::LLVM::AllocaOp allocaOp, mlir::Value bytesVal )
    {
        loc = rewriter.getUnknownLoc();    // HACK: suppress location info for these implicit memset's, so that the line
                                           // numbers in gdb don't bounce around.  The re-ordering that I now do in the
                                           // DeclareOp builder is messing things up.

        mlir::Value i8Ptr = rewriter.create<mlir::LLVM::BitcastOp>( loc, tyPtr, allocaOp );

        mlir::Value fillVal =
            rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI8, rewriter.getI8IntegerAttr( driverState.fillValue ) );

        rewriter.create<mlir::LLVM::MemsetOp>( loc, i8Ptr, fillVal, bytesVal, rewriter.getBoolAttr( false ) );
    }

    /// Lower silly::DeclareOp
    class DeclareOpLowering : public mlir::OpConversionPattern<silly::DeclareOp>
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for DeclareOpLowering
        DeclareOpLowering( mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
                           LoweringContext& loweringState, mlir::PatternBenefit benefit )
            : mlir::OpConversionPattern<silly::DeclareOp>( typeConverter, context, benefit ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::DeclareOp
        mlir::LogicalResult matchAndRewrite( silly::DeclareOp declareOp, OpAdaptor adaptor,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            mlir::Location loc = declareOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.declare: " << declareOp << '\n' );

            rewriter.setInsertionPoint( declareOp );

            silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
            mlir::Type elemType = varTy.getElementType();

            if ( !elemType.isIntOrFloat() )
            {
                return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
            }

            unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();
            unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;

            // FIXME: could pack array creation for i1 types (elemType.isInteger( 1 )).  For now, just use a
            // separate byte for each.
            unsigned alignment = lState.preferredTypeAlignment( declareOp, elemType );

            mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
            llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

            mlir::Value sizeVal;
            mlir::Value bytesVal;
            int64_t arraySize = 1;
            if ( !shape.empty() )
            {
                arraySize = shape[0];

                if ( ( arraySize <= 0 ) || ( shape.size() != 1 ) )
                {
                    return rewriter.notifyMatchFailure(
                        declareOp, llvm::formatv( "expected non-zero arraySize ({0}), and one varType dimension ({1})",
                                                  arraySize, shape.size() ) );
                }

                sizeVal = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.tyI64,
                                                                   rewriter.getI64IntegerAttr( arraySize ) );
                bytesVal = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, lState.tyI64, rewriter.getI64IntegerAttr( arraySize * elemSizeInBytes ) );
            }
            else
            {
                sizeVal = lState.getI64one( loc, rewriter );
                bytesVal = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.tyI64,
                                                                    rewriter.getI64IntegerAttr( elemSizeInBytes ) );
            }

            mlir::LLVM::AllocaOp allocaOp =
                rewriter.create<mlir::LLVM::AllocaOp>( loc, lState.tyPtr, elemType, sizeVal, alignment );

            auto init = declareOp.getInitializers();
            if ( init.size() )
            {
                mlir::Type elemType = allocaOp.getElemType();
                unsigned alignment = lState.preferredTypeAlignment( declareOp, elemType );

                if ( !shape.empty() )
                {
                    for ( size_t i = 0; i < init.size(); ++i )
                    {
                        mlir::Value iVal64 = rewriter.create<mlir::LLVM::ConstantOp>(
                            loc, lState.tyI64, rewriter.getI64IntegerAttr( static_cast<int64_t>( i ) ) );

                        mlir::IndexType indexTy = rewriter.getIndexType();
                        mlir::Value idxIndex = rewriter.create<mlir::arith::IndexCastOp>( loc, indexTy, iVal64 );

                        if ( mlir::failed( lState.generateAssignment(
                                 loc, rewriter, declareOp, init[i], elemType, allocaOp, alignment,
                                 mlir::cast<mlir::TypedValue<mlir::IndexType>>( idxIndex ) ) ) )
                        {
                            return mlir::failure();
                        }
                    }
                }
                else
                {
                    if ( init.size() != 1 )
                    {
                        return rewriter.notifyMatchFailure(
                            declareOp, llvm::formatv( "scalar initializer count: {0} is not one.", init.size() ) );
                    }

                    if ( mlir::failed( lState.generateAssignment( loc, rewriter, declareOp, init[0], elemType, allocaOp,
                                                                  alignment, mlir::TypedValue<mlir::IndexType>{} ) ) )
                    {
                        return mlir::failure();
                    }
                }
            }
            else
            {
                lState.insertFill( loc, rewriter, allocaOp, bytesVal );
            }

            std::string funcName = lookupFuncNameForOp( declareOp );
            lState.setAlloca( funcName, declareOp.getOperation(), allocaOp.getOperation() );

            // rewriter.replaceOp( declareOp, allocaOp.getResult() ); // this failed.  typeconverter didn't work as
            // hoped.
            rewriter.eraseOp( declareOp );

            return mlir::success();
        }
    };

    /// Lower silly::StringLiteralOp
    class StringLiteralOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for StringLiteralOpLowering
        StringLiteralOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context,
                                 mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::StringLiteralOp::getOperationName(), benefit, context ),
              lState( loweringState )
        {
        }

        /// Lowering workhorse for silly::StringLiteralOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::StringLiteralOp stringLiteralOp = cast<silly::StringLiteralOp>( op );
            mlir::Location loc = stringLiteralOp.getLoc();

            mlir::StringAttr strAttr = stringLiteralOp.getValueAttr();
            std::string strValue = strAttr.getValue().str();
            size_t strLen = strValue.size();

            mlir::LLVM::GlobalOp globalOp = lState.lookupOrInsertGlobalOp( loc, rewriter, strAttr, strLen );
            if ( !globalOp )
            {
                return rewriter.notifyMatchFailure( op, "Failed to create or lookup string literal global" );
            }

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    /// Lower silly::AssignOp to llvm.store (after type conversions, if required)
    class AssignOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for AssignOpLowering
        AssignOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::AssignOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::AssignOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::AssignOp assignOp = cast<silly::AssignOp>( op );
            mlir::Location loc = assignOp.getLoc();

            // silly.assign %0 : <i64[[]]> = %c1_i64 : i64 loc(#loc3)
            LLVM_DEBUG( llvm::dbgs() << "Lowering AssignOp: " << *op << '\n' );

            mlir::Value var = assignOp.getVar();
            assert( var );

            std::string funcName = lookupFuncNameForOp( op );

            silly::DeclareOp declareOp = var.getDefiningOp<silly::DeclareOp>();
            mlir::LLVM::AllocaOp allocaOp = lState.getAlloca( funcName, declareOp.getOperation() );

            LLVM_DEBUG( {
                llvm::dbgs() << "AssignOp.  module state::\n";
                mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
                mod->dump();
            } );
            assert( allocaOp );

            mlir::Value value = assignOp.getValue();

            silly::varType varTy = mlir::cast<silly::varType>( var.getType() );
            mlir::Type elemType = varTy.getElementType();
            unsigned alignment = lState.preferredTypeAlignment( op, elemType );

            if ( mlir::failed( lState.generateAssignment( loc, rewriter, op, value, elemType, allocaOp, alignment,
                                                          assignOp.getIndex() ) ) )
            {
                return mlir::failure();
            }

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    /// Lower silly::LoadOp
    class LoadOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for LoadOpLowering
        LoadOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::LoadOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::LoadOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::LoadOp loadOp = cast<silly::LoadOp>( op );
            mlir::Location loc = loadOp.getLoc();

            // %0 = silly.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.load: " << *op << '\n' );

            mlir::Value var = loadOp.getVar();
            assert( var );
            mlir::LLVM::AllocaOp allocaOp = var.getDefiningOp<mlir::LLVM::AllocaOp>();
            assert( allocaOp );
            mlir::TypedValue<mlir::IndexType> optIndex = loadOp.getIndex();

            mlir::Type elemType = allocaOp.getElemType();
            mlir::Value load;

            if ( loadOp.getResult().getType() == lState.tyPtr )
            {
                load = allocaOp.getResult();
            }
            else
            {
                if ( optIndex )
                {
                    mlir::Value indexVal = optIndex;
                    mlir::Value basePtr = allocaOp.getResult();

                    int64_t numElems = 0;
                    if ( mlir::LLVM::ConstantOp allocationBoundsConstOp =
                             allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
                    {
                        mlir::IntegerAttr intAttr =
                            mlir::dyn_cast<mlir::IntegerAttr>( allocationBoundsConstOp.getValue() );
                        numElems = intAttr.getInt();
                    }

                    if ( !numElems )
                    {
                        return rewriter.notifyMatchFailure( op, "zero elements for attempted index access" );
                    }

                    if ( mlir::arith::ConstantIndexOp constOp = indexVal.getDefiningOp<mlir::arith::ConstantIndexOp>() )
                    {
                        int64_t idx = constOp.value();
                        if ( idx < 0 || idx >= numElems )
                        {
                            return loadOp.emitError() << "static out-of-bounds array access: index " << idx
                                                      << " is out of bounds for array of size " << numElems;
                        }
                    }

                    // Cast index to i64 for LLVM GEP
                    mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>( loc, lState.tyI64, indexVal );

                    // GEP to element
                    mlir::Value elemPtr =
                        rewriter.create<mlir::LLVM::GEPOp>( loc,
                                                            basePtr.getType(),    // result type: ptr-to-elem
                                                            elemType,             // pointee type
                                                            basePtr, mlir::ValueRange{ idxI64 } );

                    // Load array element
                    load = rewriter.create<mlir::LLVM::LoadOp>( loc, elemType, elemPtr ).getResult();
                }
                else
                {
                    // Scalar load
                    load = rewriter.create<mlir::LLVM::LoadOp>( loc, elemType, allocaOp ).getResult();
                }
            }

            LLVM_DEBUG( llvm::dbgs() << "new load op: " << load << '\n' );
            rewriter.replaceOp( op, load );

            return mlir::success();
        }
    };

    /// Lower silly::CallOp
    class CallOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for CallOpLowering
        CallOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::CallOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::CallOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::CallOp callOp = cast<silly::CallOp>( op );
            mlir::Location loc = callOp.getLoc();

            // Get the callee symbol reference (stored as "callee" attribute)
            mlir::FlatSymbolRefAttr calleeAttr = callOp->getAttrOfType<mlir::FlatSymbolRefAttr>( "callee" );
            if ( !calleeAttr )
                return mlir::failure();

            // Get result types (empty for void, one type for scalar return)
            mlir::TypeRange resultTypes = callOp.getResultTypes();

            mlir::func::CallOp mlirCall =
                rewriter.create<mlir::func::CallOp>( loc, resultTypes, calleeAttr, callOp.getOperands() );

            // Replace uses correctly
            if ( !resultTypes.empty() )
            {
                // Non-void: replace the single result
                rewriter.replaceOp( op, mlirCall.getResults() );
            }
            else
            {
                // Void: erase the op (no result to replace)
                rewriter.eraseOp( op );
            }

            return mlir::success();
        }
    };

    /// Lower silly::ScopeOp
    class ScopeOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for ScopeOpLowering
        ScopeOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::ScopeOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::ScopeOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::ScopeOp scopeOp = cast<silly::ScopeOp>( op );
            mlir::Region* funcRegion = scopeOp->getParentRegion();
            if ( !funcRegion || !isa<mlir::func::FuncOp>( scopeOp->getParentOp() ) )
            {
                return rewriter.notifyMatchFailure( op, "ScopeOp must be nested in a func::FuncOp" );
            }

            mlir::Block* entryBlock = &*funcRegion->begin();
            mlir::Operation* funcTerminator = entryBlock->getTerminator();

            // Verify that the terminator is a YieldOp
            if ( !isa<silly::YieldOp>( funcTerminator ) )
            {
                return rewriter.notifyMatchFailure( op, "Expected func::FuncOp terminator to be silly::YieldOp" );
            }

            // Erase the YieldOp first to ensure only one terminator will exist
            rewriter.eraseOp( funcTerminator );

            // If ScopeOp has a non-empty body, process its operations
            if ( !scopeOp.getBody().empty() )
            {
                mlir::Block& scopeBlock = scopeOp.getBody().front();

                // Set insertion point at the end of the func entry block
                rewriter.setInsertionPointToEnd( entryBlock );

                // Process operations in the scope block
                for ( mlir::Operation& op : llvm::make_early_inc_range( scopeBlock ) )
                {
                    if ( isa<silly::ReturnOp>( op ) )
                    {
                        // Replace silly::ReturnOp with func::ReturnOp
                        rewriter.create<mlir::func::ReturnOp>( op.getLoc(), op.getOperands() );
                        rewriter.eraseOp( &op );
                    }
                    else
                    {
                        // Move other operations to the entry block
                        rewriter.moveOpBefore( &op, entryBlock, entryBlock->end() );
                    }
                }
            }

            // Erase the original ScopeOp
            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

#if 0
    // Now unused (again)
    template <class SillOpType>
    class LowerByDeletion : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        LowerByDeletion( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( SillOpType::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs() << "Lowering (by erase): " << *op << '\n' );
            rewriter.eraseOp( op );
            return mlir::success();
        }
    };
#endif

    /// Lower silly::DebugNameOp
    class DebugNameOpLowering : public mlir::OpConversionPattern<silly::DebugNameOp>
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for DebugNameOpLowering
        DebugNameOpLowering( mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
                             LoweringContext& loweringState, mlir::PatternBenefit benefit )
            : mlir::OpConversionPattern<silly::DebugNameOp>( typeConverter, context, benefit ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::DebugNameOp
        mlir::LogicalResult matchAndRewrite( silly::DebugNameOp debugNameOp, OpAdaptor adaptor,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            mlir::Value value = debugNameOp.getValue();
            mlir::Location loc = debugNameOp.getLoc();
            mlir::FileLineColLoc fileLoc = getLocation( loc );
            std::string funcName = lookupFuncNameForOp( debugNameOp );

            if ( silly::DeclareOp declareOp = value.getDefiningOp<silly::DeclareOp>() )
            {
                std::string varName = debugNameOp.getName().str();
                mlir::LLVM::AllocaOp allocaOp = lState.getAlloca( funcName, declareOp.getOperation() );
                silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
                mlir::Type elemType = varTy.getElementType();
                LLVM_DEBUG( llvm::dbgs() << "DebugNameOpLowering: elemType: " << elemType << '\n' );

                if ( !elemType.isIntOrFloat() )
                {
                    return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
                }

                unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();

                mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
                llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

                int64_t arraySize = 1;
                if ( !shape.empty() )
                {
                    arraySize = shape[0];
                }

                if ( mlir::failed( lState.constructVariableDI( fileLoc, rewriter, declareOp, varName, elemType,
                                                               elemSizeInBits, allocaOp, arraySize ) ) )
                {
                    return mlir::failure();
                }
            }
            else
            {
                std::string varName = debugNameOp.getName().str();
                mlir::StringAttr nameAttr = debugNameOp.getNameAttr();

                mlir::Type elemType = value.getType();
                unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();

                if ( mlir::failed( lState.constructInductionVariableDI( fileLoc, rewriter, debugNameOp, value, varName,
                                                                        nameAttr, elemType, elemSizeInBits,
                                                                        funcName ) ) )
                {
                    return mlir::failure();
                }
            }

            rewriter.eraseOp( debugNameOp );
            return mlir::success();
        }
    };

    /// Lower silly.print (silly::PrintOp) to a call to __silly_print.
    class PrintOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for PrintOpLowering
        PrintOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::PrintOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::PrintOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::PrintOp printOp = mlir::cast<silly::PrintOp>( op );
            mlir::Location loc = printOp.getLoc();

            std::string funcName = lookupFuncNameForOp( op );
            mlir::LLVM::AllocaOp arrayAlloca = lState.getPrintArgs( funcName );
            assert( arrayAlloca );

            auto inputs = printOp.getInputs();
            size_t numArgs = inputs.size();

            int baseFlags = 0;
            if ( mlir::arith::ConstantIntOp flagOp =
                     mlir::dyn_cast<mlir::arith::ConstantIntOp>( printOp.getFlags().getDefiningOp() ) )
            {
                baseFlags = flagOp.value();
            }

            mlir::Location argLoc = loc;
            // mlir::Location argLoc = rewriter.getUnknownLoc();

            for ( size_t i = 0; i < numArgs; ++i )
            {
                bool isLast = ( i == numArgs - 1 );
                PrintFlags pf = static_cast<PrintFlags>( baseFlags );

                if ( !isLast )
                {
                    pf = static_cast<PrintFlags>( pf | static_cast<int>( silly::PrintFlags::PRINT_FLAGS_CONTINUE ) );
                }

                mlir::Value argStruct;
                if ( mlir::failed( lState.emitPrintArgStruct( argLoc, rewriter, op, inputs[i], pf, argStruct ) ) )
                {
                    return mlir::failure();
                }

                mlir::LLVM::ConstantOp indexVal =
                    rewriter.create<mlir::LLVM::ConstantOp>( argLoc, lState.tyI64, rewriter.getI64IntegerAttr( i ) );
                mlir::LLVM::GEPOp slotPtr = rewriter.create<mlir::LLVM::GEPOp>(
                    argLoc, lState.tyPtr, lState.printArgStructTy, arrayAlloca, mlir::ValueRange{ indexVal } );

                rewriter.create<mlir::LLVM::StoreOp>( argLoc, argStruct, slotPtr );
            }

            // Final call
            mlir::LLVM::ConstantOp numArgsConst =
                rewriter.create<mlir::LLVM::ConstantOp>( argLoc, lState.tyI32, rewriter.getI32IntegerAttr( numArgs ) );

            rewriter.create<silly::CallOp>( loc, mlir::TypeRange{}, "__silly_print",
                                            mlir::ValueRange{ numArgsConst, arrayAlloca } );

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    /// Lower silly::AbortOp
    class AbortOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for AbortOpLowering
        AbortOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::AbortOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::AbortOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::AbortOp abortOp = cast<silly::AbortOp>( op );
            mlir::Location loc = abortOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.abort: " << *op << '\n' );

            lState.createAbortCall( loc, rewriter );

            rewriter.eraseOp( op );

            return mlir::success();
        }
    };

    /// Lower silly::GetOp
    class GetOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for GetOpLowering
        GetOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::GetOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::GetOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::GetOp getOp = cast<silly::GetOp>( op );
            mlir::Location loc = getOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.get: " << *op << '\n' );

            mlir::Type inputType = getOp.getValue().getType();

            mlir::Value result;
            if ( mlir::failed( lState.createGetCall( loc, rewriter, op, inputType, result ) ) )
            {
                return mlir::failure();
            }

            rewriter.replaceOp( op, result );

            return mlir::success();
        }
    };

    /// Lower silly.negate (silly::NegOp) to LLVM arithmetic.
    class NegOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for NegOpLowering
        NegOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::NegOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::NegOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::NegOp negOp = cast<silly::NegOp>( op );
            mlir::Location loc = negOp.getLoc();
            mlir::Value result = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.negate: " << *op << '\n' );

            if ( mlir::IntegerType resulti = mlir::dyn_cast<mlir::IntegerType>( result.getType() ) )
            {
                mlir::LLVM::ConstantOp zero;
                if ( mlir::failed( lState.getIzero( loc, rewriter, op, resulti.getWidth(), zero ) ) )
                {
                    return mlir::failure();
                }

                result = rewriter.create<mlir::LLVM::SubOp>( loc, zero, result );
            }
            else if ( mlir::FloatType resultf = mlir::dyn_cast<mlir::FloatType>( result.getType() ) )
            {
                mlir::LLVM::ConstantOp zero;
                if ( resultf == lState.tyF32 )
                {
                    zero = lState.getF32zero( loc, rewriter );
                }
                else if ( resultf == lState.tyF64 )
                {
                    zero = lState.getF64zero( loc, rewriter );
                }
                else
                {
                    return rewriter.notifyMatchFailure(
                        op, llvm::formatv( "Unknown floating point type in negation operation lowering: {0}",
                                           result.getType() ) );
                }

                result = rewriter.create<mlir::LLVM::FSubOp>( loc, zero, result );
            }
            else
            {
                return rewriter.notifyMatchFailure(
                    op, llvm::formatv( "Unknown type in negation operation lowering: {0}", result.getType() ) );
            }

            rewriter.replaceOp( op, result );
            return mlir::success();
        }
    };

    /// type conversions and rewriter creations for numeric binary ops.
    template <class llvmIOpType, class llvmFOpType, bool allowFloat>
    mlir::LogicalResult binaryArithOpLoweringHelper( mlir::Location loc, LoweringContext& lState, mlir::Operation* op,
                                                     mlir::Value lhs, mlir::Value rhs,
                                                     mlir::ConversionPatternRewriter& rewriter, mlir::Type resultType,
                                                     bool needsLibMathIfFloat )
    {
        if ( resultType.isIntOrIndex() )
        {
            unsigned rwidth = resultType.getIntOrFloatBitWidth();

            if ( mlir::IntegerType lTyI = mlir::dyn_cast<mlir::IntegerType>( lhs.getType() ) )
            {
                unsigned width = lTyI.getWidth();

                if ( rwidth > width )
                {
                    lhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, resultType, lhs );
                }
                else if ( rwidth < width )
                {
                    lhs = rewriter.create<mlir::LLVM::TruncOp>( loc, resultType, lhs );
                }
            }
            else if ( lState.isTypeFloat( lhs.getType() ) )
            {
                if ( allowFloat )
                {
                    lhs = rewriter.create<mlir::LLVM::FPToSIOp>( loc, resultType, lhs );
                }
                else
                {
                    return rewriter.notifyMatchFailure( op, "float types unsupported for integer binary operation" );
                }
            }

            if ( mlir::IntegerType rTyI = mlir::dyn_cast<mlir::IntegerType>( rhs.getType() ) )
            {
                unsigned width = rTyI.getWidth();

                if ( rwidth > width )
                {
                    rhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, resultType, rhs );
                }
                else if ( rwidth < width )
                {
                    rhs = rewriter.create<mlir::LLVM::TruncOp>( loc, resultType, rhs );
                }
            }
            else if ( lState.isTypeFloat( rhs.getType() ) )
            {
                if ( allowFloat )
                {
                    rhs = rewriter.create<mlir::LLVM::FPToSIOp>( loc, resultType, rhs );
                }
                else
                {
                    return rewriter.notifyMatchFailure( op, "float types unsupported for integer binary operation" );
                }
            }

            llvmIOpType result = rewriter.create<llvmIOpType>( loc, lhs, rhs );
            rewriter.replaceOp( op, result );
        }
        else if ( allowFloat )
        {
            // Floating-point addition: ensure both operands are f64.
            if ( mlir::IntegerType lTyI = mlir::dyn_cast<mlir::IntegerType>( lhs.getType() ) )
            {
                unsigned width = lTyI.getWidth();

                if ( width == 1 )
                {
                    lhs = rewriter.create<mlir::LLVM::UIToFPOp>( loc, resultType, lhs );
                }
                else
                {
                    lhs = rewriter.create<mlir::LLVM::SIToFPOp>( loc, resultType, lhs );
                }
            }
            if ( mlir::IntegerType rTyI = mlir::dyn_cast<mlir::IntegerType>( rhs.getType() ) )
            {
                unsigned width = rTyI.getWidth();

                if ( width == 1 )
                {
                    rhs = rewriter.create<mlir::LLVM::UIToFPOp>( loc, resultType, rhs );
                }
                else
                {
                    rhs = rewriter.create<mlir::LLVM::SIToFPOp>( loc, resultType, rhs );
                }
            }

            if ( needsLibMathIfFloat )
            {
                lState.markMathLibRequired();
            }

            llvmFOpType result = rewriter.create<llvmFOpType>( loc, lhs, rhs );
            rewriter.replaceOp( op, result );
        }
        else
        {
            return rewriter.notifyMatchFailure( op, "float types unsupported for integer binary operation" );
        }

        return mlir::success();
    }

    /// Lower silly::ArithBinOp
    class ArithBinOpLowering : public mlir::ConversionPattern
    {
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for ArithBinOpLowering
        ArithBinOpLowering( LoweringContext& state, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : ConversionPattern( silly::ArithBinOp::getOperationName(), benefit, context ), lState( state )
        {
        }

        /// Lowering workhorse for silly::ArithBinOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::ArithBinOp binaryOp = cast<silly::ArithBinOp>( op );
            silly::ArithBinOpKind kind = binaryOp.getKind();

            mlir::Location loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.binary: " << *op << '\n' );

            mlir::Value lhs = operands[0];
            mlir::Value rhs = operands[1];
            mlir::Type resultType = binaryOp.getResult().getType();

            switch ( kind )
            {
                case silly::ArithBinOpKind::Add:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::AddOp, mlir::LLVM::FAddOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }
                case silly::ArithBinOpKind::Sub:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::SubOp, mlir::LLVM::FSubOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }
                case silly::ArithBinOpKind::Mul:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::MulOp, mlir::LLVM::FMulOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Div:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::SDivOp, mlir::LLVM::FDivOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Mod:
                {
                    // LLVM_DEBUG( llvm::dbgs() << "Lowering mod: " << *op << ' ' << lhs << ' ' << rhs << '\n' );
                    return binaryArithOpLoweringHelper<mlir::LLVM::SRemOp, mlir::LLVM::FRemOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, true );
                }

                // mlir::LLVM::FAddOp is a dummy operation below, knowing that it will not ever be used:
                case silly::ArithBinOpKind::And:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::AndOp, mlir::LLVM::FAddOp, false>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Or:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::OrOp, mlir::LLVM::FAddOp, false>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Xor:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::XOrOp, mlir::LLVM::FAddOp, false>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }
            }

            llvm_unreachable( "unknown arith binop kind" );

            return mlir::failure();
        }
    };

    /// A helper function for silly::ArithBinOp
    template <class IOpType, class FOpType, mlir::LLVM::ICmpPredicate ICmpPredS, mlir::LLVM::ICmpPredicate ICmpPredU,
              mlir::LLVM::FCmpPredicate FCmpPred>
    mlir::LogicalResult binaryCompareOpLoweringHelper( mlir::Location loc, LoweringContext& lState, mlir::Operation* op,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::ConversionPatternRewriter& rewriter )
    {
        mlir::IntegerType lTyI = mlir::dyn_cast<mlir::IntegerType>( lhs.getType() );
        mlir::IntegerType rTyI = mlir::dyn_cast<mlir::IntegerType>( rhs.getType() );
        mlir::FloatType lTyF = mlir::dyn_cast<mlir::FloatType>( lhs.getType() );
        mlir::FloatType rTyF = mlir::dyn_cast<mlir::FloatType>( rhs.getType() );

        if ( lTyI && rTyI )
        {
            unsigned lwidth = lTyI.getWidth();
            unsigned rwidth = rTyI.getWidth();
            mlir::LLVM::ICmpPredicate pred = ICmpPredS;

            if ( rwidth > lwidth )
            {
                if ( lwidth == 1 )
                {
                    lhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, rTyI, lhs );
                }
                else
                {
                    lhs = rewriter.create<mlir::LLVM::SExtOp>( loc, rTyI, lhs );
                }
            }
            else if ( rwidth < lwidth )
            {
                if ( rwidth == 1 )
                {
                    rhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, lTyI, rhs );
                }
                else
                {
                    rhs = rewriter.create<mlir::LLVM::SExtOp>( loc, lTyI, rhs );
                }
            }
            else if ( ( rwidth == lwidth ) && ( rwidth == 1 ) )
            {
                pred = ICmpPredU;
            }

            IOpType cmp = rewriter.create<IOpType>( loc, pred, lhs, rhs );
            rewriter.replaceOp( op, cmp.getResult() );
        }
        else if ( lTyF && rTyF )
        {
            unsigned lwidth = lTyF.getWidth();
            unsigned rwidth = rTyF.getWidth();

            if ( lwidth < rwidth )
            {
                lhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, rTyF, lhs );
            }
            else if ( rwidth < lwidth )
            {
                rhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, lTyF, rhs );
            }

            FOpType cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
            rewriter.replaceOp( op, cmp.getResult() );
        }
        else
        {
            // convert integer type to float
            if ( lTyI && rTyF )
            {
                if ( lTyI == lState.tyI1 )
                {
                    lhs = rewriter.create<mlir::arith::UIToFPOp>( loc, rTyF, lhs );
                }
                else
                {
                    lhs = rewriter.create<mlir::arith::SIToFPOp>( loc, rTyF, lhs );
                }
            }
            else if ( rTyI && lTyF )
            {
                if ( rTyI == lState.tyI1 )
                {
                    rhs = rewriter.create<mlir::arith::UIToFPOp>( loc, lTyF, rhs );
                }
                else
                {
                    rhs = rewriter.create<mlir::arith::SIToFPOp>( loc, lTyF, rhs );
                }
            }
            else
            {
                return rewriter.notifyMatchFailure( op, "Unsupported type combination" );
            }

            FOpType cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
            rewriter.replaceOp( op, cmp.getResult() );
        }

        return mlir::success();
    }

    /// Lower silly::CmpBinOp
    class CmpBinOpLowering : public mlir::ConversionPattern
    {
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for CmpBinOpLowering
        CmpBinOpLowering( LoweringContext& state, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : ConversionPattern( silly::CmpBinOp::getOperationName(), benefit, context ), lState( state )
        {
        }

        /// Lowering workhorse for silly::CmpBinOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::CmpBinOp binaryOp = cast<silly::CmpBinOp>( op );
            silly::CmpBinOpKind kind = binaryOp.getKind();

            mlir::Location loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.cmp: " << *op << '\n' );

            mlir::Value lhs = operands[0];
            mlir::Value rhs = operands[1];

            switch ( kind )
            {
                case silly::CmpBinOpKind::Less:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::slt, mlir::LLVM::ICmpPredicate::ult,
                                                         mlir::LLVM::FCmpPredicate::olt>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }
                case silly::CmpBinOpKind::LessEq:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::sle, mlir::LLVM::ICmpPredicate::ule,
                                                         mlir::LLVM::FCmpPredicate::ole>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }
                case silly::CmpBinOpKind::Equal:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::eq, mlir::LLVM::ICmpPredicate::eq,
                                                         mlir::LLVM::FCmpPredicate::oeq>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }

                case silly::CmpBinOpKind::NotEqual:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::ne, mlir::LLVM::ICmpPredicate::ne,
                                                         mlir::LLVM::FCmpPredicate::one>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }
            }

            llvm_unreachable( "unknown arith binop kind" );

            return mlir::failure();
        }
    };

    /// Orchestrate the lowering of the Silly dialect.
    ///
    /// When this is done, if successful, we will be left with LLVM mlir dialect Ops
    /// and a couple standalone mlir dialect Ops (func.func, module, ...)
    class SillyToLLVMLoweringPass
        : public mlir::PassWrapper<SillyToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( SillyToLLVMLoweringPass )

        /// lowering pass.  squirrel away the DriverState for later use.
        SillyToLLVMLoweringPass( silly::DriverState* pst ) : pDriverState{ pst }
        {
        }

        /// load dependent dialects
        void getDependentDialects( mlir::DialectRegistry& registry ) const override
        {
            registry.insert<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
        }

        /// do the lowering
        void runOnOperation() override
        {
            mlir::ModuleOp mod = getOperation();
            LLVM_DEBUG( {
                llvm::dbgs() << "Starting SillyToLLVMLoweringPass on:\n";
                mod->dump();
            } );

            LoweringContext lState( mod, *pDriverState );
            lState.createDICompileUnit();

            for ( mlir::func::FuncOp funcOp : mod.getBodyRegion().getOps<mlir::func::FuncOp>() )
            {
                LLVM_DEBUG( {
                    llvm::dbgs() << "Generating !DISubroutineType() for mlir::func::FuncOp: " << funcOp.getSymName()
                                 << "\n";
                } );

                bool error = lState.createPerFuncState( funcOp );
                if ( error )
                {
                    LLVM_DEBUG( llvm::dbgs()
                                << "!DISubroutineType() creation for " << funcOp.getSymName() << " failed" );
                    signalPassFailure();
                    return;
                }
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After createPerFuncState:\n";
                mod->dump();
            } );

            // First phase: Lower silly operations except ScopeOp and YieldOp
            {
                mlir::ConversionTarget target( getContext() );
                target.addLegalDialect<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect, silly::SillyDialect,
                                       mlir::scf::SCFDialect>();
                target.addIllegalOp<silly::AssignOp, silly::DeclareOp, silly::LoadOp, silly::NegOp, silly::PrintOp,
                                    silly::GetOp, silly::StringLiteralOp, silly::AbortOp, silly::DebugNameOp,
                                    silly::ArithBinOp, silly::CmpBinOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp,
                                  silly::ScopeOp, silly::YieldOp, silly::ReturnOp, silly::CallOp, mlir::func::CallOp,
                                  mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::YieldOp>();

                mlir::RewritePatternSet patterns( &getContext() );
                patterns.add<AssignOpLowering, LoadOpLowering, NegOpLowering, PrintOpLowering, AbortOpLowering,
                             GetOpLowering, StringLiteralOpLowering, ArithBinOpLowering, CmpBinOpLowering>(
                    lState, &getContext(), 1 );

                patterns.add<DeclareOpLowering, DebugNameOpLowering>( lState.getTypeConverter(), &getContext(), lState,
                                                                      1 );

                if ( failed( applyFullConversion( mod, target, std::move( patterns ) ) ) )
                {
                    LLVM_DEBUG( llvm::dbgs() << "Silly Lowering: First phase failed\n" );
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG( {
                    llvm::dbgs() << "After first phase (silly ops lowered):\n";
                    mod->dump();
                } );
            }

            // Second phase: Inline ScopeOp and erase YieldOp
            {
                mlir::ConversionTarget target( getContext() );
                target.addLegalDialect<mlir::LLVM::LLVMDialect>();
                target.addIllegalOp<silly::ScopeOp, silly::YieldOp, silly::ReturnOp, silly::CallOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp>();
                target.addIllegalDialect<mlir::scf::SCFDialect>();
                target.addIllegalDialect<mlir::cf::ControlFlowDialect>();    // forces lowering

                mlir::RewritePatternSet patterns( &getContext() );
                patterns.add<CallOpLowering, ScopeOpLowering>( lState, &getContext(), 1 );

                // SCF -> CF
                mlir::populateSCFToControlFlowConversionPatterns( patterns );

                mlir::arith::populateArithToLLVMConversionPatterns( lState.getTypeConverter(), patterns );

                // CF -> LLVM
                mlir::cf::populateControlFlowToLLVMConversionPatterns( lState.getTypeConverter(), patterns );

                if ( failed( applyFullConversion( mod, target, std::move( patterns ) ) ) )
                {
                    LLVM_DEBUG( llvm::dbgs() << "Silly Lowering: Second phase failed\n" );
                    signalPassFailure();
                    return;
                }
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After successful SillyToLLVMLoweringPass:\n";
                for ( mlir::Operation& op : mod->getRegion( 0 ).front() )
                {
                    op.dump();
                }
            } );
        }

       private:
        silly::DriverState*
            pDriverState;    ///< stuff from the driver (is debug enabled, ...)  Also mark when -lm will be required.
    };

}    // namespace silly

namespace mlir
{
    /// Silly dialect pass framework
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass()
    {
        return createSillyToLLVMLoweringPass( nullptr );    // Default to no optimization
    }

    /// Silly dialect pass framework
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass( silly::DriverState* pst )
    {
        return std::make_unique<silly::SillyToLLVMLoweringPass>( pst );
    }

    /// Custom registration with bool parameter
    void registerSillyToLLVMLoweringPass( silly::DriverState* pst )
    {
        ::mlir::registerPass( [pst]() -> std::unique_ptr<::mlir::Pass>
                              { return mlir::createSillyToLLVMLoweringPass( pst ); } );
    }
}    // namespace mlir

// vim: et ts=4 sw=4
