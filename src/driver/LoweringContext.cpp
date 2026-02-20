///
/// @file    LoweringContext.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This file implements helper functions for silly dialect to LLVM-IR lowering.
///
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <llvm/Support/Path.h>

#include <format>

#include "DriverState.hpp"
#include "ModuleInsertionPointGuard.hpp"
#include "LoweringContext.hpp"
#include "helper.hpp"

/// --debug- type for lowering
#define DEBUG_TYPE "silly-lowering-context"

/// For llvm.ident
#define COMPILER_VERSION " V9"

namespace silly
{
    LoweringContext::LoweringContext( mlir::ModuleOp& moduleOp, silly::DriverState& ds )
        : driverState{ ds }, mod{ moduleOp }, builder{ mod.getRegion() }, typeConverter{ builder.getContext() }
    {
        mlir::MLIRContext* context = builder.getContext();
        typ.initialize( builder, context );

#if 0    // tried to use this in DeclareOp and DebugNameOp lowering, but it didn't work.  Revisit this later.
        typeConverter.addConversion( []( silly::varType type ) -> mlir::Type
                                     { return mlir::LLVM::LLVMPointerType::get( type.getContext() ); } );
#endif

        printArgStructTy =
            mlir::LLVM::LLVMStructType::getLiteral( context,
                                                    {
                                                        typ.i32,    // kind: PrintKind (i32)
                                                        typ.i32,    // flags: PrintFlags (i32)
                                                        typ.i64,    // i, or string length, or bitcast double
                                                        typ.ptr     // ptr: const char* (only used for STRING)
                                                    },
                                                    /*isPacked=*/false );
    }

    unsigned LoweringContext::preferredTypeAlignment( mlir::Operation* op, mlir::Type elemType )
    {
        mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
        assert( mod );
        mlir::DataLayout dataLayout( mod );
        unsigned alignment = dataLayout.getTypePreferredAlignment( elemType );

        return alignment;
    }

    void LoweringContext::createSillyPrintPrototype()
    {
        if ( !printFunc )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), { typ.i32, typ.ptr }, {} );

            printFunc = builder.create<mlir::func::FuncOp>( mod.getLoc(), "__silly_print", funcType );
            printFunc.setVisibility( mlir::SymbolTable::Visibility::Private );
        }
    }

    void LoweringContext::createSillyAbortPrototype()
    {
        if ( !printFuncAbort )
        {
            ModuleInsertionPointGuard ip( mod, builder );

            mlir::FunctionType funcType = mlir::FunctionType::get( builder.getContext(), { typ.i64, typ.ptr, typ.i32 }, {} );
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
        createSillyGetPrototype( getFuncI1, typ.i8, "__silly_get_i1" );
    }

    inline void LoweringContext::createSillyGetI8Prototype()
    {
        createSillyGetPrototype( getFuncI8, typ.i8, "__silly_get_i8" );
    }

    inline void LoweringContext::createSillyGetI16Prototype()
    {
        createSillyGetPrototype( getFuncI16, typ.i16, "__silly_get_i16" );
    }

    inline void LoweringContext::createSillyGetI32Prototype()
    {
        createSillyGetPrototype( getFuncI32, typ.i32, "__silly_get_i32" );
    }

    inline void LoweringContext::createSillyGetI64Prototype()
    {
        createSillyGetPrototype( getFuncI64, typ.i64, "__silly_get_i64" );
    }

    inline void LoweringContext::createSillyGetF32Prototype()
    {
        createSillyGetPrototype( getFuncF32, typ.f32, "__silly_get_f32" );
    }

    inline void LoweringContext::createSillyGetF64Prototype()
    {
        createSillyGetPrototype( getFuncF64, typ.f64, "__silly_get_f64" );
    }

    void LoweringContext::createDICompileUnit( mlir::Location loc )
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

            std::string filename = filenameFromLoc( loc );

            // Construct module level DI state:
            fileAttr = mlir::LLVM::DIFileAttr::get( context, filename, "." );
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
                mlir::FileLineColLoc loc = locationToFLCLoc( funcLoc );
                unsigned line = loc.getLine();
                unsigned scopeLine = line;

                // Get the location of the First operation in the block for the scopeLine:
                if ( !entryBlock.empty() )
                {
                    mlir::Operation* firstOp = &entryBlock.front();
                    mlir::Location firstLoc = firstOp->getLoc();
                    mlir::FileLineColLoc scopeLoc = locationToFLCLoc( firstLoc );

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

        if ( maxPrintArgs )
        {
            mlir::OpBuilder::InsertionGuard guard(builder);

            mlir::Location loc = builder.getUnknownLoc();

            mlir::Region& funcRegion = funcOp.getBody();

            mlir::Block* entryBlock = &funcRegion.front();
            assert( entryBlock );    // Is this ever not the case?

            if ( !entryBlock )
            {
                return true;
            }

            builder.setInsertionPointToStart( entryBlock );

            funcState[funcName].printArgs = builder.create<mlir::LLVM::AllocaOp>(
                loc, typ.ptr, printArgStructTy,
                builder.create<mlir::LLVM::ConstantOp>( loc, typ.i64, builder.getI64IntegerAttr( maxPrintArgs ) ) );
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

    mlir::LogicalResult LoweringContext::constructVariableDI( mlir::FileLineColLoc fileLoc,
                                                              mlir::ConversionPatternRewriter& rewriter,
                                                              mlir::Operation* op, llvm::StringRef varName,
                                                              mlir::Type& elemType, unsigned elemSizeInBits,
                                                              mlir::Value value, int64_t arraySize )
    {
        if ( !driverState.wantDebug )
        {
            return mlir::success();
        }

        assert( fileLoc );

        std::string funcName = lookupFuncNameForOp( op );
        mlir::LLVM::DISubprogramAttr sub = funcState[funcName].subProgramDI;
        assert( sub );

        mlir::LLVM::DILocalVariableAttr diVar;
        mlir::LLVM::DITypeAttr diType;

        const char* typeName;
        unsigned dwType;
        unsigned elemStorageSizeInBits;

        if ( mlir::failed( LoweringContext::infoForVariableDI( fileLoc, rewriter, op, varName, elemType, elemSizeInBits,
                                                               arraySize, typeName, dwType, elemStorageSizeInBits ) ) )
        {
            return mlir::failure();
        }

        mlir::MLIRContext* context = builder.getContext();

        if ( mlir::LLVM::AllocaOp allocaOp = value.getDefiningOp<mlir::LLVM::AllocaOp>() )
        {
            allocaOp->setAttr( "bindc_name", builder.getStringAttr( varName ) );

            unsigned totalSizeInBits = elemStorageSizeInBits * arraySize;
            if ( arraySize > 1 )
            {
                // Create base type for array elements
                mlir::LLVM::DIBasicTypeAttr baseType = mlir::LLVM::DIBasicTypeAttr::get(
                    context, llvm::dwarf::DW_TAG_base_type, builder.getStringAttr( typeName ), elemStorageSizeInBits,
                    dwType );

                // Create subrange for array (count = arraySize, lowerBound = 0)
                mlir::IntegerAttr countAttr = mlir::IntegerAttr::get( typ.i64, arraySize );
                mlir::IntegerAttr lowerBoundAttr = mlir::IntegerAttr::get( typ.i64, 0 );
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
                context, sub, builder.getStringAttr( varName ), fileAttr, fileLoc.getLine(),
                /*argNo=*/0, totalSizeInBits, diType, mlir::LLVM::DIFlags::Zero );

            builder.setInsertionPointAfter( allocaOp );
            builder.create<mlir::LLVM::DbgDeclareOp>( fileLoc, allocaOp, diVar );
        }
        else
        {
            mlir::LLVM::DITypeAttr diType =
                mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                  builder.getStringAttr( typeName ), elemStorageSizeInBits, dwType );

            mlir::LLVM::DILocalVariableAttr diVar = mlir::LLVM::DILocalVariableAttr::get(
                context, sub, builder.getStringAttr( varName ), fileAttr, fileLoc.getLine(),
                /*argNo=*/0, elemStorageSizeInBits, diType, mlir::LLVM::DIFlags::Zero );

            // Emit llvm.dbg.value
            // Empty expression for direct value binding
            mlir::LLVM::DIExpressionAttr emptyExpr = mlir::LLVM::DIExpressionAttr::get( context, {} );

            rewriter.create<mlir::LLVM::DbgValueOp>( fileLoc, value, diVar, emptyExpr );
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

            mlir::LLVM::LLVMArrayType arrayType = mlir::LLVM::LLVMArrayType::get( typ.i8, strLen );

            mlir::SmallVector<char> stringData( stringLit.begin(), stringLit.end() );
            mlir::DenseElementsAttr denseAttr =
                mlir::DenseElementsAttr::get( mlir::RankedTensorType::get( { static_cast<int64_t>( strLen ) }, typ.i8 ),
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
        mlir::FileLineColLoc fileLoc = locationToFLCLoc( loc );

        const std::string& filename = fileLoc.getFilename().str();
        mlir::StringAttr strAttr;
        if ( driverState.abortOmitPath )
        {
            llvm::StringRef name = llvm::sys::path::filename( filename );
            strAttr = builder.getStringAttr( name );
        }
        else
        {
            strAttr = builder.getStringAttr( filename );
        }

        std::string strValue = strAttr.getValue().str();
        size_t strLen = strValue.size();

        mlir::LLVM::ConstantOp sizeConst =
            rewriter.create<mlir::LLVM::ConstantOp>( loc, typ.i64, rewriter.getI64IntegerAttr( strLen ) );

        mlir::LLVM::GlobalOp globalOp = lookupOrInsertGlobalOp( loc, rewriter, strAttr, strLen );
        mlir::Value input = rewriter.create<mlir::LLVM::AddressOfOp>( loc, globalOp );

        mlir::LLVM::ConstantOp lineConst =
            rewriter.create<mlir::LLVM::ConstantOp>( loc, typ.i32, rewriter.getI32IntegerAttr( fileLoc.getLine() ) );

        createSillyAbortPrototype();
        const char* name = "__silly_abort";
        rewriter.create<mlir::func::CallOp>( loc, mlir::TypeRange{}, name,
                                             mlir::ValueRange{ sizeConst, input, lineConst } );
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
                valuePayload = rewriter.create<mlir::LLVM::ZExtOp>( loc, typ.i64, input );
            }
            else if ( intTy.getWidth() < 64 )
            {
                valuePayload = rewriter.create<mlir::LLVM::SExtOp>( loc, typ.i64, input );
            }
            else
            {
                valuePayload = input;
            }
        }
        else if ( mlir::FloatType floatTy = mlir::dyn_cast<mlir::FloatType>( inputType ) )
        {
            kind = PrintKind::F64;
            if ( inputType == typ.f32 )
            {
                valuePayload = rewriter.create<mlir::LLVM::FPExtOp>( loc, typ.f64, input );
            }
            else
            {
                valuePayload = input;
            }

            valuePayload = rewriter.create<mlir::LLVM::BitcastOp>( loc, typ.i64, valuePayload );
        }
        else if ( inputType == typ.ptr )
        {
            kind = PrintKind::STRING;

            int64_t numElems = 0;
            mlir::Value ptr = input;

            if ( silly::LoadOp loadOp = input.getDefiningOp<silly::LoadOp>() )
            {
                mlir::Value var = loadOp.getVar();
                assert( var );

                std::string funcName = lookupFuncNameForOp( op );
                silly::DeclareOp declareOp = var.getDefiningOp<silly::DeclareOp>();
                mlir::LLVM::AllocaOp allocaOp = getAlloca( funcName, declareOp.getOperation() );
                assert( allocaOp );

                if ( allocaOp.getElemType() != typ.i8 )
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
                rewriter.create<mlir::LLVM::ConstantOp>( loc, typ.i64, rewriter.getI64IntegerAttr( numElems ) );
            strPtr = ptr;
        }
        else
        {
            return rewriter.notifyMatchFailure( op, "unsupported print argument type" );
        }

        // Insert kind (index 0)
        mlir::LLVM::ConstantOp kindVal = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, typ.i32, rewriter.getI32IntegerAttr( static_cast<uint32_t>( kind ) ) );
        structVal = rewriter.create<mlir::LLVM::InsertValueOp>( loc, printArgStructTy, structVal, kindVal, 0 );

        // Insert flags (index 1)
        mlir::LLVM::ConstantOp flagsVal = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, typ.i32, rewriter.getI32IntegerAttr( static_cast<uint32_t>( flags ) ) );
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
            mlir::Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>( loc, typ.ptr );

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
                    inputType = typ.i8;
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
            if ( inputType == typ.f32 )
            {
                name = "__silly_get_f32";
                createSillyGetF32Prototype();
            }
            else if ( inputType == typ.f64 )
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

        mlir::func::CallOp callOp =
            rewriter.create<mlir::func::CallOp>( loc, mlir::TypeRange{ inputType }, name, mlir::ValueRange{} );
        mlir::Value result = callOp.getResult( 0 );

        if ( isBool )
        {
            result = rewriter.create<mlir::LLVM::TruncOp>( loc, typ.i1, result );
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
        if ( valType == typ.f64 )
        {
            if ( mlir::isa<mlir::IntegerType>( elemType ) )
            {
                value = rewriter.create<mlir::LLVM::FPToSIOp>( loc, elemType, value );
            }
            else if ( elemType == typ.f32 )
            {
                value = rewriter.create<mlir::LLVM::FPTruncOp>( loc, elemType, value );
            }
        }
        else if ( valType == typ.f32 )
        {
            if ( mlir::isa<mlir::IntegerType>( elemType ) )
            {
                value = rewriter.create<mlir::LLVM::FPToSIOp>( loc, elemType, value );
            }
            else if ( elemType == typ.f64 )
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
            if ( elemType != typ.i8 )
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
                rewriter.create<mlir::LLVM::ConstantOp>( loc, typ.i64, rewriter.getI64IntegerAttr( copySize ) );

            rewriter.create<mlir::LLVM::MemcpyOp>( loc, destPtr, globalPtr, sizeConst, rewriter.getBoolAttr( false ) );

            // If target array is larger than string literal, zero out the remaining bytes
            if ( numElems > (int64_t)literalStrLen )
            {
                // Compute the offset: destPtr + literalStrLen
                mlir::LLVM::ConstantOp offsetConst =
                    rewriter.create<mlir::LLVM::ConstantOp>( loc, typ.i64, rewriter.getI64IntegerAttr( literalStrLen ) );
                mlir::LLVM::GEPOp destPtrOffset = rewriter.create<mlir::LLVM::GEPOp>(
                    loc, destPtr.getType(), elemType, destPtr, mlir::ValueRange{ offsetConst } );

                // Compute the number of bytes to zero: numElems - literalStrLen
                mlir::LLVM::ConstantOp remainingSize = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, typ.i64, rewriter.getI64IntegerAttr( numElems - literalStrLen ) );

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
            mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>( loc, typ.i64, indexVal );

            mlir::Type elemPtrTy = destBasePtr.getType();

            mlir::Value elemPtr = rewriter.create<mlir::LLVM::GEPOp>( loc,
                                                                      elemPtrTy,    // result type
                                                                      elemType,     // pointee type
                                                                      destBasePtr, mlir::ValueRange{ idxI64 } );

            // Nice to have (untested): Runtime bounds check -- make this a compile option?
            // if (numElems > 0) {
            //     mlir::Value sizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
            //         loc, typ.i64, rewriter.getI64IntegerAttr(numElems));
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

        mlir::Value i8Ptr = rewriter.create<mlir::LLVM::BitcastOp>( loc, typ.ptr, allocaOp );

        mlir::Value fillVal =
            rewriter.create<mlir::LLVM::ConstantOp>( loc, typ.i8, rewriter.getI8IntegerAttr( driverState.fillValue ) );

        rewriter.create<mlir::LLVM::MemsetOp>( loc, i8Ptr, fillVal, bytesVal, rewriter.getBoolAttr( false ) );
    }

    void LoweringContext::markMathLibRequired()
    {
        driverState.needsMathLib = true;
    }
}    // namespace silly
