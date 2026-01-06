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
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <format>
#include <numeric>

#include "SillyExceptions.hpp"
#include "lowering.hpp"
#include "loweringContext.hpp"

#define DEBUG_TYPE "silly-lowering"

// for llvm.ident and DICompileUnitAttr
#define COMPILER_NAME "silly"

/// For llvm.ident
#define COMPILER_VERSION " V8"

namespace silly
{
    class ModuleInsertionPointGuard
    {
        mlir::OpBuilder& builder;
        mlir::OpBuilder::InsertPoint oldIP;

       public:
        ModuleInsertionPointGuard( mlir::ModuleOp& mod, mlir::OpBuilder& opBuilder )
            : builder{ opBuilder }, oldIP{ builder.saveInsertionPoint() }
        {
            builder.setInsertionPointToStart( mod.getBody() );
        }

        ~ModuleInsertionPointGuard()
        {
            builder.restoreInsertionPoint( oldIP );
        }
    };

    mlir::FileLineColLoc getLocation( mlir::Location loc )
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

    LoweringContext::LoweringContext( mlir::ModuleOp& moduleOp, const silly::DriverState& ds )
        : driverState{ ds }, mod{ moduleOp }, builder{ mod.getRegion() }, typeConverter{ builder.getContext() }
    {
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
    /// Throws an exception for unsupported widths.
    inline mlir::LLVM::ConstantOp LoweringContext::getIzero( mlir::Location loc,
                                                             mlir::ConversionPatternRewriter& rewriter, unsigned width )
    {
        switch ( width )
        {
            case 8:
                return getI8zero( loc, rewriter );
            case 16:
                return getI16zero( loc, rewriter );
            case 32:
                return getI32zero( loc, rewriter );
            case 64:
                return getI64zero( loc, rewriter );
        }

        throw ExceptionWithContext( __FILE__, __LINE__, __func__, std::format( "Unexpected integer size {}", width ) );
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

    mlir::LLVM::ConstantOp LoweringContext::getFzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter,
                                                      unsigned width )
    {
        switch ( width )
        {
            case 32:
                return getF32zero( loc, rewriter );
            case 64:
                return getF64zero( loc, rewriter );
        }

        throw ExceptionWithContext( __FILE__, __LINE__, __func__, std::format( "Unexpected float size {}", width ) );
    }

    template <class Ty>
    void LoweringContext::createSillyPrintPrototype( mlir::func::FuncOp& printOp, Ty type, const char* name )
    {
        if ( !printOp )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), { type }, {} );

            printOp = builder.create<mlir::func::FuncOp>( mod.getLoc(), name, funcType );
            printOp.setVisibility( mlir::SymbolTable::Visibility::Private );
        }
    }

    inline void LoweringContext::createSillyPrintF64Prototype()
    {
        createSillyPrintPrototype( printFuncF64, tyF64, "__silly_print_f64" );
    }

    inline void LoweringContext::createSillyPrintI64Prototype()
    {
        createSillyPrintPrototype( printFuncI64, tyI64, "__silly_print_i64" );
    }

    void LoweringContext::createSillyPrintStringPrototype()
    {
        if ( !printFuncString )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), { tyI64, tyPtr }, {} );
            printFuncString = builder.create<mlir::func::FuncOp>( mod.getLoc(), "__silly_print_string", funcType );
            printFuncString.setVisibility( mlir::SymbolTable::Visibility::Private );
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

    void LoweringContext::createFuncDebug( mlir::func::FuncOp funcOp )
    {
        if ( driverState.wantDebug )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::MLIRContext* context = builder.getContext();
            std::string funcName = funcOp.getSymName().str();

            mlir::LLVM::DISubroutineTypeAttr subprogramType = createDISubroutineType( funcOp );

            mlir::Location funcLoc = funcOp.getLoc();
            mlir::FileLineColLoc loc = getLocation( funcLoc );
            unsigned line = loc.getLine();
            unsigned scopeLine = line;

            mlir::Region &region = funcOp.getRegion();

            mlir::Block &entryBlock = region.front();

            // Get the location of the First operation in the block for the scopeLine:
            if (!entryBlock.empty()) {
              mlir::Operation *firstOp = &entryBlock.front();
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

            subprogramAttr[funcName] = sub;
        }
    }

    std::string LoweringContext::lookupFuncNameForOp( mlir::Operation* op )
    {
        mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

        return funcOp.getSymName().str();
    }

    mlir::LLVM::AllocaOp LoweringContext::lookupLocalSymbolReference( mlir::Operation* op, const std::string& varName )
    {
        mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

        LLVM_DEBUG( {
            llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
            funcOp->dump();
        } );

        std::string funcName = funcOp->getName().getStringRef().str();

        std::string funcNameAndVarName = funcName + "::" + varName;

        mlir::Operation* alloca = symbolToAlloca[funcNameAndVarName];
        return mlir::dyn_cast<mlir::LLVM::AllocaOp>( alloca );
    }

    void LoweringContext::createLocalSymbolReference( mlir::LLVM::AllocaOp allocaOp, const std::string& varName )
    {
        mlir::func::FuncOp funcOp = getEnclosingFuncOp( allocaOp );
        std::string funcName = funcOp->getName().getStringRef().str();

        std::string funcNameAndVarName = funcName + "::" + varName;

        symbolToAlloca[funcNameAndVarName] = allocaOp;
    }

    void LoweringContext::constructVariableDI( llvm::StringRef varName, mlir::Type& elemType, mlir::FileLineColLoc loc,
                                               unsigned elemSizeInBits, mlir::LLVM::AllocaOp& allocaOp,
                                               int64_t arraySize )
    {
        if ( driverState.wantDebug )
        {
            mlir::MLIRContext* context = builder.getContext();

            allocaOp->setAttr( "bindc_name", builder.getStringAttr( varName ) );

            mlir::LLVM::DILocalVariableAttr diVar;
            mlir::LLVM::DITypeAttr diType;

            const char* typeName{};
            unsigned dwType = llvm::dwarf::DW_ATE_signed;
            unsigned elemStorageSizeInBits = elemSizeInBits;    // Storage size (e.g., i1 uses i8)

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
                        llvm_unreachable( "Unsupported integer type size" );
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
                        llvm_unreachable( "Unsupported float type size" );
                    }
                }
            }
            else
            {
                llvm_unreachable( "Unsupported type for debug info" );
            }

            std::string funcName = lookupFuncNameForOp( allocaOp );
            mlir::LLVM::DISubprogramAttr sub = subprogramAttr[funcName];
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

        createLocalSymbolReference( allocaOp, varName.str() );
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

    mlir::LLVM::GlobalOp LoweringContext::lookupOrInsertGlobalOp( mlir::ConversionPatternRewriter& rewriter,
                                                                  mlir::StringAttr& stringLit, mlir::Location loc,
                                                                  size_t strLen )
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

    silly::CallOp LoweringContext::createPrintCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc,
                                                    mlir::Value input )
    {
        mlir::Type inputType = input.getType();
        silly::CallOp result;

        if ( mlir::IntegerType inputi = mlir::dyn_cast<mlir::IntegerType>( inputType ) )
        {
            unsigned width = inputi.getWidth();

            if ( width == 1 )
            {
                input = rewriter.create<mlir::LLVM::ZExtOp>( loc, tyI64, input );
            }
            else if ( width < 64 )
            {
                input = rewriter.create<mlir::LLVM::SExtOp>( loc, tyI64, input );
            }

            createSillyPrintI64Prototype();
            result = rewriter.create<silly::CallOp>( loc, mlir::TypeRange{}, "__silly_print_i64",
                                                     mlir::ValueRange{ input } );
        }
        else if ( mlir::FloatType inputf = mlir::dyn_cast<mlir::FloatType>( inputType ) )
        {
            if ( inputType == tyF32 )
            {
                input = rewriter.create<mlir::LLVM::FPExtOp>( loc, tyF64, input );
            }
            else
            {
                assert( inputType == tyF64 );
            }

            createSillyPrintF64Prototype();
            result = rewriter.create<silly::CallOp>( loc, mlir::TypeRange{}, "__silly_print_f64",
                                                     mlir::ValueRange{ input } );
        }
        else if ( inputType == tyPtr )
        {
            // Find AllocaOp for size and element type
            int64_t numElems = 0;
            if ( silly::LoadOp loadOp = input.getDefiningOp<silly::LoadOp>() )
            {
                mlir::SymbolRefAttr varNameAttr = loadOp.getVarName();
                assert( varNameAttr );

                // Get string (e.g., "x")
                std::string varName = varNameAttr.getLeafReference().str();
                LLVM_DEBUG( { llvm::dbgs() << "LoadOp variable name: " << varName << "\n"; } );

                mlir::LLVM::AllocaOp allocaOp = lookupLocalSymbolReference( loadOp, varName );

                // Validate element type is i8
                mlir::Type elemType = allocaOp.getElemType();
                assert( elemType == tyI8 );

                if ( mlir::LLVM::ConstantOp constOp = allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
                {
                    mlir::IntegerAttr intAttr = mlir::dyn_cast<mlir::IntegerAttr>( constOp.getValue() );
                    numElems = intAttr.getInt();
                }
            }
            else if ( silly::StringLiteralOp stringLitOp = input.getDefiningOp<silly::StringLiteralOp>() )
            {
                mlir::StringAttr strAttr = stringLitOp.getValueAttr();
                llvm::StringRef strValue = strAttr.getValue();
                numElems = strValue.size();

                mlir::LLVM::GlobalOp globalOp = lookupGlobalOp( strAttr );
                input = rewriter.create<mlir::LLVM::AddressOfOp>( loc, globalOp );
            }
            else
            {
                LLVM_DEBUG( {
                    llvm::errs() << "why am I here?\n";
                    input.dump();
                    // mlir::ModuleOp mod = input.getParentOfType<mlir::ModuleOp>();
                    // mod.dump();
                } );

                assert( 0 );    // should not get here.
            }
            // numElems == 0 is okay here.  t/c: shortstring2.silly

            mlir::LLVM::ConstantOp sizeConst =
                rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( numElems ) );

            createSillyPrintStringPrototype();
            const char* name = "__silly_print_string";
            result =
                rewriter.create<silly::CallOp>( loc, mlir::TypeRange{}, name, mlir::ValueRange{ sizeConst, input } );
        }
        else
        {
            assert( 0 );    // Error: unsupported type
        }

#if 0
            LLVM_DEBUG( {
                llvm::errs() << "######################### mod dump after print lowering\n";
                mlir::ModuleOp mod = result->getParentOfType<mlir::ModuleOp>();
                mod.dump();
            } );
#endif

        return result;
    }

    mlir::Value LoweringContext::createGetCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc,
                                                mlir::Type inputType )
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
                    assert( 0 && "Unexpected integer size" );
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
                assert( 0 && "Unexpected floating point type" );
            }
        }
        else
        {
            assert( 0 && "Error: unsupported type" );
        }

        silly::CallOp callOp =
            rewriter.create<silly::CallOp>( loc, mlir::TypeRange{ inputType }, name, mlir::ValueRange{} );
        mlir::Value result = *callOp.getResult().begin();

        if ( isBool )
        {
            result = rewriter.create<mlir::LLVM::TruncOp>( loc, tyI1, result );
        }

        return result;
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

            // Get DISubprogram from subprogramAttr
            mlir::LLVM::DISubprogramAttr sub = subprogramAttr[funcName];
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

    class DeclareOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        DeclareOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::DeclareOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::DeclareOp declareOp = cast<silly::DeclareOp>( op );
            mlir::Location loc = declareOp.getLoc();
            bool param = declareOp.isParameter();

            //   silly.declare "x" : i32
            LLVM_DEBUG( llvm::dbgs() << std::format( "Lowering silly.declare: param: {}", param ) << declareOp
                                     << '\n' );

            rewriter.setInsertionPoint( op );

            mlir::StringRef varName = declareOp.getName();
            mlir::Type elemType = declareOp.getType();

            if ( param )
            {
                mlir::IntegerAttr paramNumberAttr = declareOp.getParamNumberAttr();
                if ( !paramNumberAttr )
                {
                    return rewriter.notifyMatchFailure( op, "Parameter missing param_number attribute" );
                }
                int64_t paramIndex = paramNumberAttr.getInt();
                mlir::func::FuncOp funcOp = op->getParentOfType<mlir::func::FuncOp>();
                if ( paramIndex >= funcOp.getNumArguments() )
                {
                    return rewriter.notifyMatchFailure( op, "Parameter index out of bounds" );
                }
                mlir::Value value = funcOp.getArgument( paramIndex );
                std::string funcName = funcOp.getSymName().str();

                unsigned alignment = lState.preferredTypeAlignment( funcOp, elemType );

                // Allocate stack space for the parameter
                mlir::Value one = lState.getI64one( loc, rewriter );
                mlir::LLVM::AllocaOp allocaOp =
                    rewriter.create<mlir::LLVM::AllocaOp>( loc, lState.tyPtr, elemType, one, alignment );
                allocaOp->setAttr( "bindc_name", rewriter.getStringAttr( varName + ".addr" ) );

                // Store the parameter value in the allocated memory
                rewriter.create<mlir::LLVM::StoreOp>( loc, value, allocaOp );

                lState.constructParameterDI( getLocation( loc ), rewriter, varName.str(), allocaOp, elemType,
                                             paramIndex, funcName );

                lState.createLocalSymbolReference( allocaOp, varName.str() );
            }
            else
            {
                if ( !elemType.isIntOrFloat() )
                {
                    return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
                }

                unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();
                // unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;

                // FIXME: could pack array creation for i1 types (elemType.isInteger( 1 )).  For now, just use a
                // separate byte for each.
                unsigned alignment = lState.preferredTypeAlignment( op, elemType );

                mlir::Value sizeVal;
                int64_t arraySize = 1;
                if ( declareOp.getSize().has_value() )
                {
                    arraySize = declareOp.getSize().value();
                    if ( arraySize <= 0 )
                    {
                        return rewriter.notifyMatchFailure( declareOp, "array size must be positive" );
                    }
                    sizeVal = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.tyI64,
                                                                       rewriter.getI64IntegerAttr( arraySize ) );
                }
                else
                {
                    sizeVal = lState.getI64one( loc, rewriter );
                }

                mlir::LLVM::AllocaOp allocaOp =
                    rewriter.create<mlir::LLVM::AllocaOp>( loc, lState.tyPtr, elemType, sizeVal, alignment );
                lState.constructVariableDI( varName, elemType, getLocation( loc ), elemSizeInBits, allocaOp,
                                            arraySize );
            }

            rewriter.eraseOp( op );

            return mlir::success();
        }
    };

    class StringLiteralOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        StringLiteralOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context,
                                 mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::StringLiteralOp::getOperationName(), benefit, context ),
              lState( loweringState )
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::StringLiteralOp stringLiteralOp = cast<silly::StringLiteralOp>( op );
            mlir::Location loc = stringLiteralOp.getLoc();

            mlir::StringAttr strAttr = stringLiteralOp.getValueAttr();
            std::string strValue = strAttr.getValue().str();
            size_t strLen = strValue.size();

            mlir::LLVM::GlobalOp globalOp = lState.lookupOrInsertGlobalOp( rewriter, strAttr, loc, strLen );
            if ( !globalOp )
            {
                return rewriter.notifyMatchFailure( op, "Failed to create or lookup string literal global" );
            }

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    // Lower AssignOp to llvm.store (after type conversions, if required)
    class AssignOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        AssignOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::AssignOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::AssignOp assignOp = cast<silly::AssignOp>( op );
            mlir::Location loc = assignOp.getLoc();

            // (ins StrAttr:$name, AnyType:$value);
            // silly.assign "x", %0 : i32
            LLVM_DEBUG( llvm::dbgs() << "Lowering AssignOp: " << *op << '\n' );

            mlir::SymbolRefAttr varNameAttr = assignOp.getVarName();
            assert( varNameAttr );

            // Get string (e.g., "x")
            std::string varName = varNameAttr.getLeafReference().str();
            LLVM_DEBUG( { llvm::dbgs() << "AssignOp variable name: " << varName << "\n"; } );

            mlir::LLVM::AllocaOp allocaOp = lState.lookupLocalSymbolReference( assignOp, varName );

            mlir::Value value = assignOp.getValue();
            mlir::Type valType = value.getType();

            // varName: i1v
            // value: %true = arith.constant true
            // valType: i1
            LLVM_DEBUG( llvm::dbgs() << "varName: " << varName << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "value: " << value << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "valType: " << valType << '\n' );

            // extract parameters from the allocaOp so we know what to do here:
            mlir::Type elemType = allocaOp.getElemType();
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
                value = lState.castToElemType( loc, rewriter, value, valType, elemType );

                unsigned alignment = lState.preferredTypeAlignment( op, elemType );
                rewriter.create<mlir::LLVM::StoreOp>( loc, value, allocaOp, alignment );
            }
            else if ( silly::StringLiteralOp stringLitOp = value.getDefiningOp<silly::StringLiteralOp>() )
            {
                if ( elemType != lState.tyI8 )
                {
                    return rewriter.notifyMatchFailure( assignOp, "string assignment requires i8 array" );
                }
                if ( numElems == 0 )
                {
                    return rewriter.notifyMatchFailure( assignOp, "invalid array size" );
                }

                mlir::StringAttr strAttr = stringLitOp.getValueAttr();
                llvm::StringRef strValue = strAttr.getValue();
                size_t literalStrLen = strValue.size();
                mlir::LLVM::GlobalOp globalOp = lState.lookupGlobalOp( strAttr );

                mlir::LLVM::AddressOfOp globalPtr = rewriter.create<mlir::LLVM::AddressOfOp>( loc, globalOp );

                mlir::Value destPtr = allocaOp.getResult();

                int copySize = std::min( (int)numElems, (int)literalStrLen );
                mlir::LLVM::ConstantOp sizeConst = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, lState.tyI64, rewriter.getI64IntegerAttr( copySize ) );

                rewriter.create<mlir::LLVM::MemcpyOp>( loc, destPtr, globalPtr, sizeConst,
                                                       rewriter.getBoolAttr( false ) );

                // If target array is larger than string literal, zero out the remaining bytes
                if ( numElems > (int64_t)literalStrLen )
                {
                    // Compute the offset: destPtr + literalStrLen
                    mlir::LLVM::ConstantOp offsetConst = rewriter.create<mlir::LLVM::ConstantOp>(
                        loc, lState.tyI64, rewriter.getI64IntegerAttr( literalStrLen ) );
                    mlir::LLVM::GEPOp destPtrOffset = rewriter.create<mlir::LLVM::GEPOp>(
                        loc, destPtr.getType(), elemType, destPtr, mlir::ValueRange{ offsetConst } );

                    // Compute the number of bytes to zero: numElems - literalStrLen
                    mlir::LLVM::ConstantOp remainingSize = rewriter.create<mlir::LLVM::ConstantOp>(
                        loc, lState.tyI64, rewriter.getI64IntegerAttr( numElems - literalStrLen ) );

                    // Set remaining bytes to zero
                    rewriter.create<mlir::LLVM::MemsetOp>( loc, destPtrOffset, lState.getI8zero( loc, rewriter ),
                                                           remainingSize, rewriter.getBoolAttr( false ) );
                }
            }
            else    // ARRAY ELEMENT or UNSUPPORTED ASSIGNMENT
            {
                mlir::TypedValue<mlir::IndexType> optIndex = assignOp.getIndex();    // std::optional<Value>

                if ( !optIndex )
                {
                    // Assigning a non-string-literal to an array (e.g., t = some_expr;)
                    // This is not supported (arrays are not first-class values)
                    return rewriter.notifyMatchFailure(
                        assignOp, "assignment of non-string-literal to array variable without index is not supported" );
                }

                mlir::Value indexVal = optIndex;
                mlir::Value destBasePtr = allocaOp.getResult();

                assert( numElems && "non-scalar, non-string assignment must be an array with non-zero size" );
                if ( mlir::arith::ConstantIndexOp constOp = indexVal.getDefiningOp<mlir::arith::ConstantIndexOp>() )
                {
                    int64_t idx = constOp.value();
                    if ( idx < 0 || idx >= numElems )
                    {
                        return assignOp.emitError() << "static out-of-bounds array access: index " << idx
                                                    << " is out of bounds for array of size " << numElems;
                    }
                }

                // Cast index to i64 for LLVM dialect GEP indexing
                mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>( loc, lState.tyI64, indexVal );

                mlir::Type elemPtrTy = destBasePtr.getType();

                mlir::Value elemPtr = rewriter.create<mlir::LLVM::GEPOp>( loc,
                                                                          elemPtrTy,    // result type
                                                                          elemType,     // pointee type
                                                                          destBasePtr, mlir::ValueRange{ idxI64 } );

                // Nice to have (untested): Runtime bounds check -- make this a compile option?
                // if (numElems > 0) {
                //     mlir::Value sizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
                //         loc, lState.tyI64, rewriter.getI64IntegerAttr(numElems));
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

                value = lState.castToElemType( loc, rewriter, value, valType, elemType );

                unsigned alignment = lState.preferredTypeAlignment( op, elemType );
                rewriter.create<mlir::LLVM::StoreOp>( loc, value, elemPtr, alignment );
            }

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    // Lower LessOp, ... (after type conversions, if required)
    template <class SillyOp, class IOpType, class FOpType, mlir::LLVM::ICmpPredicate ICmpPredS,
              mlir::LLVM::ICmpPredicate ICmpPredU, mlir::LLVM::FCmpPredicate FCmpPred>
    class ComparisonOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        ComparisonOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( SillyOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            SillyOp compareOp = cast<SillyOp>( op );
            mlir::Location loc = compareOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering ComparisonOp: " << *op << '\n' );

            mlir::Value lhs = compareOp.getLhs();
            mlir::Value rhs = compareOp.getRhs();

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
    };

    using LessOpLowering =
        ComparisonOpLowering<silly::LessOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::slt,
                             mlir::LLVM::ICmpPredicate::ult, mlir::LLVM::FCmpPredicate::olt>;

    using LessEqualOpLowering =
        ComparisonOpLowering<silly::LessEqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::sle,
                             mlir::LLVM::ICmpPredicate::ule, mlir::LLVM::FCmpPredicate::ole>;

    using EqualOpLowering =
        ComparisonOpLowering<silly::EqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::eq,
                             mlir::LLVM::ICmpPredicate::eq, mlir::LLVM::FCmpPredicate::oeq>;

    using NotEqualOpLowering =
        ComparisonOpLowering<silly::NotEqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::ne,
                             mlir::LLVM::ICmpPredicate::ne, mlir::LLVM::FCmpPredicate::one>;

    class LoadOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        LoadOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::LoadOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::LoadOp loadOp = cast<silly::LoadOp>( op );
            mlir::Location loc = loadOp.getLoc();

            // %0 = silly.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.load: " << *op << '\n' );

            std::string varName = loadOp.getVarNameAttr().getRootReference().getValue().str();
            mlir::LLVM::AllocaOp allocaOp = lState.lookupLocalSymbolReference( loadOp, varName );
            mlir::TypedValue<mlir::IndexType> optIndex = loadOp.getIndex();

            LLVM_DEBUG( llvm::dbgs() << "varName: " << varName << '\n' );

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

                    assert( numElems );

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

    class CallOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        CallOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::CallOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

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

    class ScopeOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        ScopeOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::ScopeOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

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

    // Lower silly.print to a call to __silly_print.
    class PrintOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        PrintOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::PrintOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
#if 0
            mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
#endif

            silly::PrintOp printOp = cast<silly::PrintOp>( op );
            mlir::Location loc = printOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.print: " << *op << '\n' );

            bool done{};
            for ( mlir::Value input : printOp.getInputs() )
            {
                assert( !done && "Only one input supported in silly.print lowering" );
                LLVM_DEBUG( llvm::dbgs() << "input: " << input << '\n' );

                silly::CallOp result = lState.createPrintCall( rewriter, loc, input );

                rewriter.replaceOp( op, result );
                done = true;
            }

#if 0
            LLVM_DEBUG( {
                llvm::errs() << "######################### mod dump after PrintOpLowering::matchAndRewrite replaceOp\n";
                mod.dump();
            } );
#endif

            return mlir::success();
        }
    };

    class GetOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        GetOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::GetOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::GetOp getOp = cast<silly::GetOp>( op );
            mlir::Location loc = getOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.get: " << *op << '\n' );

            mlir::Type inputType = getOp.getValue().getType();

            mlir::Value result = lState.createGetCall( rewriter, loc, inputType );

            rewriter.replaceOp( op, result );

            return mlir::success();
        }
    };

    // Lower silly.negate to LLVM arithmetic.
    class NegOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        NegOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::NegOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::NegOp negOp = cast<silly::NegOp>( op );
            mlir::Location loc = negOp.getLoc();
            mlir::Value result = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.negate: " << *op << '\n' );

            if ( mlir::IntegerType resulti = mlir::dyn_cast<mlir::IntegerType>( result.getType() ) )
            {
                result = rewriter.create<mlir::LLVM::SubOp>( loc, lState.getIzero( loc, rewriter, resulti.getWidth() ),
                                                             result );
            }
            else if ( mlir::FloatType resultf = mlir::dyn_cast<mlir::FloatType>( result.getType() ) )
            {
                unsigned w{};
                if ( resultf == lState.tyF32 )
                {
                    w = 32;
                }
                else if ( resultf == lState.tyF64 )
                {
                    w = 64;
                }

                result = rewriter.create<mlir::LLVM::FSubOp>( loc, lState.getFzero( loc, rewriter, w ), result );
            }
            else
            {
                llvm_unreachable( "Unknown type in negation operation lowering." );
            }

            rewriter.replaceOp( op, result );
            return mlir::success();
        }
    };

    // Lower silly.binary to LLVM arithmetic.
    template <class SillyBinaryOpType, class llvmIOpType, class llvmFOpType, bool allowFloat>
    class BinaryOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        BinaryOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( SillyBinaryOpType::getOperationName(), benefit, context ),
              lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            SillyBinaryOpType binaryOp = cast<SillyBinaryOpType>( op );
            mlir::Location loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.binary: " << *op << '\n' );

            mlir::Type resultType = binaryOp.getResult().getType();

            mlir::Value lhs = operands[0];
            mlir::Value rhs = operands[1];
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
                        llvm_unreachable( "float types unsupported for integer binary operation" );
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
                        llvm_unreachable( "float types unsupported for integer binary operation" );
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
                llvmFOpType result = rewriter.create<llvmFOpType>( loc, lhs, rhs );
                rewriter.replaceOp( op, result );
            }
            else
            {
                llvm_unreachable( "float types unsupported for integer binary operation" );
            }

            return mlir::success();
        }
    };

    using AddOpLowering = BinaryOpLowering<silly::AddOp, mlir::LLVM::AddOp, mlir::LLVM::FAddOp, true>;
    using SubOpLowering = BinaryOpLowering<silly::SubOp, mlir::LLVM::SubOp, mlir::LLVM::FSubOp, true>;
    using MulOpLowering = BinaryOpLowering<silly::MulOp, mlir::LLVM::MulOp, mlir::LLVM::FMulOp, true>;
    using DivOpLowering = BinaryOpLowering<silly::DivOp, mlir::LLVM::SDivOp, mlir::LLVM::FDivOp, true>;

    // mlir::LLVM::FAddOp is a dummy operation here, knowing that it will not ever be used:
    using XorOpLowering = BinaryOpLowering<silly::XorOp, mlir::LLVM::XOrOp, mlir::LLVM::FAddOp, false>;
    using AndOpLowering = BinaryOpLowering<silly::AndOp, mlir::LLVM::AndOp, mlir::LLVM::FAddOp, false>;
    using OrOpLowering = BinaryOpLowering<silly::OrOp, mlir::LLVM::OrOp, mlir::LLVM::FAddOp, false>;

    // Lower arith.constant to LLVM constant.
    class ConstantOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        ConstantOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( mlir::arith::ConstantOp::getOperationName(), benefit, context ),
              lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            mlir::arith::ConstantOp constantOp = cast<mlir::arith::ConstantOp>( op );
            mlir::Location loc = constantOp.getLoc();
            mlir::TypedAttr valueAttr = constantOp.getValue();

            LLVM_DEBUG( llvm::dbgs() << "Lowering arith.constant: " << *op << '\n' );

            if ( mlir::FloatAttr fAttr = dyn_cast<mlir::FloatAttr>( valueAttr ) )
            {
                mlir::LLVM::ConstantOp value = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.tyF64, fAttr );
                rewriter.replaceOp( op, value );
                return mlir::success();
            }
            else if ( mlir::IntegerAttr intAttr = dyn_cast<mlir::IntegerAttr>( valueAttr ) )
            {
                mlir::LLVM::ConstantOp value = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.tyI64, intAttr );
                rewriter.replaceOp( op, value );
                return mlir::success();
            }

            return mlir::failure();
        }
    };

    class SillyToLLVMLoweringPass
        : public mlir::PassWrapper<SillyToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( SillyToLLVMLoweringPass )

        SillyToLLVMLoweringPass( silly::DriverState* pst ) : pDriverState{ pst }
        {
        }

        void getDependentDialects( mlir::DialectRegistry& registry ) const override
        {
            registry.insert<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
        }

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
                lState.createFuncDebug( funcOp );
            }

            // First phase: Lower silly operations except ScopeOp and YieldOp
            {
                mlir::ConversionTarget target( getContext() );
                target.addLegalDialect<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect, silly::SillyDialect,
                                       mlir::scf::SCFDialect>();
                target.addIllegalOp<silly::AddOp, silly::AndOp, silly::AssignOp, silly::DeclareOp, silly::DivOp,
                                    silly::EqualOp, silly::LessEqualOp, silly::LessOp, silly::LoadOp, silly::MulOp,
                                    silly::NegOp, silly::NotEqualOp, silly::OrOp, silly::PrintOp, silly::GetOp,
                                    silly::StringLiteralOp, silly::SubOp, silly::XorOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp,
                                  silly::ScopeOp, silly::YieldOp, silly::ReturnOp, silly::CallOp, mlir::func::CallOp,
                                  mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::YieldOp>();

                mlir::RewritePatternSet patterns( &getContext() );
                patterns.add<AddOpLowering, AndOpLowering, AssignOpLowering, ConstantOpLowering, DeclareOpLowering,
                             DivOpLowering, EqualOpLowering, LessEqualOpLowering, LessOpLowering, LoadOpLowering,
                             MulOpLowering, NegOpLowering, NotEqualOpLowering, OrOpLowering, PrintOpLowering,
                             GetOpLowering, StringLiteralOpLowering, SubOpLowering, XorOpLowering>( lState,
                                                                                                    &getContext(), 1 );

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
        silly::DriverState* pDriverState;
    };

}    // namespace silly

namespace mlir
{
    // Parameterless version for TableGen
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass()
    {
        return createSillyToLLVMLoweringPass( nullptr );    // Default to no optimization
    }

    // Parameterized version
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass( silly::DriverState* pst )
    {
        return std::make_unique<silly::SillyToLLVMLoweringPass>( pst );
    }

    // Custom registration with bool parameter
    void registerSillyToLLVMLoweringPass( silly::DriverState* pst )
    {
        ::mlir::registerPass( [pst]() -> std::unique_ptr<::mlir::Pass>
                              { return mlir::createSillyToLLVMLoweringPass( pst ); } );
    }
}    // namespace mlir

// vim: et ts=4 sw=4
