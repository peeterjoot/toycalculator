/**
 * @file    lowering.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   This file implements the LLVM-IR lowering pattern matching operators
 */
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
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>    // For future multi-function support.
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>    // For FileLineColLoc
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <format>
#include <numeric>

#include "ToyDialect.h"
#include "lowering.h"

#define DEBUG_TYPE "toy-lowering"

#define ENTRY_SYMBOL_NAME "main"
#define COMPILER_NAME "toycalculator"
#define COMPILER_VERSION " V2"

using namespace mlir;

namespace toy
{
    mlir::FileLineColLoc getLocation( mlir::Location loc )
    {
        // Cast Location to FileLineColLoc
        auto fileLineLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        assert( fileLineLoc );

        return fileLineLoc;
    }

    class loweringContext
    {
       private:
        // Caching these may not be a good idea, as they are created with a single loc value, but using an existing
        // constant is also allowed, so maybe that's okay?
        mlir::LLVM::ConstantOp pr_one_I64;
        mlir::LLVM::ConstantOp pr_zero_F64;
        mlir::LLVM::ConstantOp pr_zero_I32;

        mlir::LLVM::DIFileAttr pr_fileAttr;
        mlir::LLVM::DISubprogramAttr pr_subprogramAttr;
        std::unordered_map<std::string, mlir::LLVM::GlobalOp> pr_stringLiterals;
        mlir::LLVM::LLVMFuncOp pr_mainFunc;
        mlir::LLVM::LLVMFuncOp pr_printFuncF64;
        mlir::LLVM::LLVMFuncOp pr_printFuncI64;
        mlir::LLVM::LLVMFuncOp pr_printFuncString;

        const toy::driverState& pr_driverState;
        ModuleOp& pr_module;
        OpBuilder pr_builder;

       public:
        DenseMap<llvm::StringRef, mlir::LLVM::AllocaOp> symbolToAlloca;

        loweringContext( ModuleOp& module, const toy::driverState& driverState )
            : pr_driverState{ driverState }, pr_module{ module }, pr_builder{ module.getRegion() }
        {
        }

        OpBuilder& getBuilder()
        {
            return pr_builder;
        }

        mlir::LLVM::ConstantOp getI64one( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_one_I64 )
            {
                pr_one_I64 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr( 1 ) );
            }

            return pr_one_I64;
        }

        mlir::LLVM::ConstantOp getI32zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_I32 )
            {
                pr_zero_I32 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr( 0 ) );
            }

            return pr_zero_I32;
        }

        mlir::LLVM::ConstantOp getF64zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_F64 )
            {
                pr_zero_F64 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getF64Type(), rewriter.getF64FloatAttr( 0 ) );
            }

            return pr_zero_F64;
        }

        // Set data_layout,ident,target_triple:
        void setModuleAttrs()
        {
#if 0    // Oops: don't really need these.  Already doing this in driver.cpp for the assembly printer (at the LLVM level
         // after all lowering and translation)
            std::string targetTriple = llvm::sys::getDefaultTargetTriple();
            llvm::Triple triple( targetTriple );
            assert( triple.isArch64Bit() && triple.isOSLinux() );

            std::string error;
            const llvm::Target* target = llvm::TargetRegistry::lookupTarget( targetTriple, error );
            assert( target );
            llvm::TargetOptions options;
            auto targetMachine = std::unique_ptr<llvm::TargetMachine>( target->createTargetMachine(
                targetTriple, "generic", "", options, std::optional<llvm::Reloc::Model>( llvm::Reloc::PIC_ ) ) );
            assert( targetMachine );
            std::string dataLayoutStr = targetMachine->createDataLayout().getStringRepresentation();

            pr_module->setAttr( "llvm.data_layout", pr_builder.getStringAttr( dataLayoutStr ) );
            pr_module->setAttr( "llvm.target_triple", pr_builder.getStringAttr( targetTriple ) );
#endif
            pr_module->setAttr( "llvm.ident", pr_builder.getStringAttr( COMPILER_NAME COMPILER_VERSION ) );
        }

        mlir::LLVM::LLVMFuncOp toyPrintF64()
        {
            if ( !pr_printFuncF64 )
            {
                auto oldIP = pr_builder.saveInsertionPoint();
                pr_builder.setInsertionPointToStart( pr_module.getBody() );

                auto ctx = pr_builder.getContext();
                auto pr_printFuncF64Type =
                    LLVM::LLVMFunctionType::get( LLVM::LLVMVoidType::get( ctx ), { pr_builder.getF64Type() }, false );
                pr_printFuncF64 = pr_builder.create<LLVM::LLVMFuncOp>( pr_module.getLoc(), "__toy_print_f64",
                                                                       pr_printFuncF64Type, LLVM::Linkage::External );
                pr_builder.restoreInsertionPoint( oldIP );
            }

            return pr_printFuncF64;
        }

        mlir::LLVM::LLVMFuncOp toyPrintI64()
        {
            if ( !pr_printFuncI64 )
            {
                auto oldIP = pr_builder.saveInsertionPoint();
                pr_builder.setInsertionPointToStart( pr_module.getBody() );

                auto ctx = pr_builder.getContext();
                auto printFuncI64Type =
                    LLVM::LLVMFunctionType::get( LLVM::LLVMVoidType::get( ctx ), { pr_builder.getI64Type() }, false );
                pr_printFuncI64 = pr_builder.create<LLVM::LLVMFuncOp>( pr_module.getLoc(), "__toy_print_i64",
                                                                       printFuncI64Type, LLVM::Linkage::External );

                pr_builder.restoreInsertionPoint( oldIP );
            }

            return pr_printFuncI64;
        }

        mlir::LLVM::LLVMFuncOp toyPrintString()
        {
            if ( !pr_printFuncString )
            {
                auto oldIP = pr_builder.saveInsertionPoint();
                pr_builder.setInsertionPointToStart( pr_module.getBody() );

                auto ctx = pr_builder.getContext();
                auto ptrType = LLVM::LLVMPointerType::get( ctx );
                auto printFuncStringType = LLVM::LLVMFunctionType::get( LLVM::LLVMVoidType::get( ctx ),
                                                                        { pr_builder.getI64Type(), ptrType }, false );
                pr_printFuncString = pr_builder.create<LLVM::LLVMFuncOp>(
                    pr_module.getLoc(), "__toy_print_string", printFuncStringType, LLVM::Linkage::External );

                pr_builder.restoreInsertionPoint( oldIP );
            }

            return pr_printFuncString;
        }

        // Create an entry block in the "main" function
        Block* addEntryBlock( ConversionPatternRewriter& rewriter )
        {
            Block* entryBlock = pr_mainFunc.addEntryBlock( rewriter );
            rewriter.setInsertionPointToStart( entryBlock );
            return entryBlock;
        }

        void createMain()
        {
            auto ctx = pr_builder.getContext();
            auto mainFuncType = LLVM::LLVMFunctionType::get( pr_builder.getI32Type(), {}, false );
            pr_mainFunc = pr_builder.create<LLVM::LLVMFuncOp>( pr_module.getLoc(), ENTRY_SYMBOL_NAME, mainFuncType,
                                                               LLVM::Linkage::External );

            if ( pr_driverState.wantDebug )
            {
                // Construct pr_module level DI state:
                pr_fileAttr = mlir::LLVM::DIFileAttr::get( ctx, pr_driverState.filename, "." );
                auto distinctAttr = mlir::DistinctAttr::create( pr_builder.getUnitAttr() );
                auto compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
                    ctx, distinctAttr, llvm::dwarf::DW_LANG_C, pr_fileAttr, pr_builder.getStringAttr( COMPILER_NAME ),
                    false, mlir::LLVM::DIEmissionKind::Full, mlir::LLVM::DINameTableKind::Default );
                auto ta = mlir::LLVM::DIBasicTypeAttr::get( ctx, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                            pr_builder.getStringAttr( "int" ), 32,
                                                            (unsigned)llvm::dwarf::DW_ATE_signed );
                llvm::SmallVector<mlir::LLVM::DITypeAttr, 1> typeArray;
                typeArray.push_back( ta );
                auto subprogramType = mlir::LLVM::DISubroutineTypeAttr::get( ctx, 0, typeArray );
                pr_subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                    ctx, mlir::DistinctAttr::create( pr_builder.getUnitAttr() ), compileUnitAttr, pr_fileAttr,
                    pr_builder.getStringAttr( ENTRY_SYMBOL_NAME ), pr_builder.getStringAttr( ENTRY_SYMBOL_NAME ),
                    pr_fileAttr, 1, 1, mlir::LLVM::DISubprogramFlags::Definition, subprogramType,
                    llvm::ArrayRef<mlir::LLVM::DINodeAttr>{}, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{} );
                pr_mainFunc->setAttr( "llvm.debug.subprogram", pr_subprogramAttr );

                // This is the key to ensure that translateModuleToLLVMIR does not strip the location info (instead
                // converts loc's into !dbg's)
                pr_mainFunc->setLoc( pr_builder.getFusedLoc( { pr_module.getLoc() }, pr_subprogramAttr ) );
            }
        }

        void constructVariableDI( llvm::StringRef varName, mlir::Type& elemType, mlir::FileLineColLoc loc,
                                  unsigned elemSizeInBits, mlir::LLVM::AllocaOp& allocaOp, int64_t arraySize = 1 )
        {
            auto ctx = pr_builder.getContext();
            if ( pr_driverState.wantDebug )
            {
                allocaOp->setAttr( "bindc_name", pr_builder.getStringAttr( varName ) );

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
                            typeName = "char";    // Use "char" for STRING arrays
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

                unsigned totalSizeInBits = elemStorageSizeInBits * arraySize;
                if ( arraySize > 1 )
                {
                    // Create base type for array elements
                    auto baseType = mlir::LLVM::DIBasicTypeAttr::get( ctx, llvm::dwarf::DW_TAG_base_type,
                                                                      pr_builder.getStringAttr( typeName ),
                                                                      elemStorageSizeInBits, dwType );

                    // Create subrange for array (count = arraySize, lowerBound = 0)
                    auto countAttr = mlir::IntegerAttr::get( mlir::IntegerType::get( ctx, 64 ), arraySize );
                    auto lowerBoundAttr = mlir::IntegerAttr::get( mlir::IntegerType::get( ctx, 64 ), 0 );
                    auto subrange = mlir::LLVM::DISubrangeAttr::get( ctx, countAttr, lowerBoundAttr,
                                                                     /*upperBound=*/nullptr, /*stride=*/nullptr );

                    // Create array type
                    auto alignInBits = elemStorageSizeInBits;    // Alignment matches element size
                    diType = mlir::LLVM::DICompositeTypeAttr::get(
                        ctx, llvm::dwarf::DW_TAG_array_type, pr_builder.getStringAttr( "" ), pr_fileAttr,
                        /*line=*/0, pr_subprogramAttr, baseType, mlir::LLVM::DIFlags::Zero, totalSizeInBits,
                        alignInBits, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{ subrange },
                        /*dataLocation=*/nullptr, /*rank=*/nullptr, /*allocated=*/nullptr, /*associated=*/nullptr );
                }
                else
                {
                    // Scalar type
                    diType = mlir::LLVM::DIBasicTypeAttr::get( ctx, llvm::dwarf::DW_TAG_base_type,
                                                               pr_builder.getStringAttr( typeName ),
                                                               elemStorageSizeInBits, dwType );
                }

                diVar = mlir::LLVM::DILocalVariableAttr::get(
                    ctx, pr_subprogramAttr, pr_builder.getStringAttr( varName ), pr_fileAttr, loc.getLine(),
                    /*argNo=*/0, totalSizeInBits, diType, mlir::LLVM::DIFlags::Zero );

                pr_builder.setInsertionPointAfter( allocaOp );
                pr_builder.create<mlir::LLVM::DbgDeclareOp>( loc, allocaOp, diVar );
            }

            symbolToAlloca[varName] = allocaOp;
        }

        mlir::LLVM::GlobalOp lookupGlobalOp( mlir::StringAttr& stringLit )
        {
            mlir::LLVM::GlobalOp globalOp;
            auto it = pr_stringLiterals.find( stringLit.str() );
            if ( it != pr_stringLiterals.end() )
            {
                globalOp = it->second;
                LLVM_DEBUG( llvm::dbgs() << std::format( "Found global: {} for string literal '{}'\n",
                                                         globalOp.getSymName().str(), stringLit.str() ) );
            }

            return globalOp;
        }

        mlir::LLVM::GlobalOp lookupOrInsertGlobalOp( ConversionPatternRewriter& rewriter, mlir::StringAttr& stringLit,
                                                     mlir::Location loc, size_t copySize, size_t strLen )
        {
            mlir::LLVM::GlobalOp globalOp;
            auto it = pr_stringLiterals.find( stringLit.str() );
            if ( it != pr_stringLiterals.end() )
            {
                globalOp = it->second;
                LLVM_DEBUG( llvm::dbgs() << "Reusing global: " << globalOp.getSymName() << '\n' );
            }
            else
            {
                auto savedIP = rewriter.saveInsertionPoint();
                rewriter.setInsertionPointToStart( pr_module.getBody() );

                auto i8Type = rewriter.getI8Type();
                auto arrayType = mlir::LLVM::LLVMArrayType::get( i8Type, copySize );

                SmallVector<char> stringData( stringLit.begin(), stringLit.end() );
                if ( copySize > strLen )
                {
                    stringData.push_back( '\0' );
                }
                auto denseAttr =
                    DenseElementsAttr::get( RankedTensorType::get( { static_cast<int64_t>( copySize ) }, i8Type ),
                                            ArrayRef<char>( stringData ) );

                std::string globalName = "str_" + std::to_string( pr_stringLiterals.size() );
                globalOp = rewriter.create<LLVM::GlobalOp>( loc, arrayType, true, LLVM::Linkage::Private, globalName,
                                                            denseAttr );
                globalOp->setAttr( "unnamed_addr", rewriter.getUnitAttr() );

                pr_stringLiterals[stringLit.str()] = globalOp;
                LLVM_DEBUG( llvm::dbgs() << "Created global: " << globalName << '\n' );

                rewriter.restoreInsertionPoint( savedIP );
            }

            return globalOp;
        }

        mlir::LLVM::CallOp createPrintCall( ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::Value input )
        {
            mlir::Type inputType = input.getType();
            mlir::LLVM::CallOp result;

            if ( auto inputi = mlir::dyn_cast<IntegerType>( inputType ) )
            {
                auto width = inputi.getWidth();

                if ( width == 1 )
                {
                    input = rewriter.create<mlir::LLVM::ZExtOp>( loc, rewriter.getI64Type(), input );
                }
                else if ( width < 64 )
                {
                    input = rewriter.create<mlir::LLVM::SExtOp>( loc, rewriter.getI64Type(), input );
                }

                result = rewriter.create<LLVM::CallOp>( loc, toyPrintI64(), ValueRange{ input } );
            }
            else if ( auto inputf = mlir::dyn_cast<FloatType>( inputType ) )
            {
                if ( inputType.isF32() )
                {
                    input = rewriter.create<LLVM::FPExtOp>( loc, rewriter.getF64Type(), input );
                }
                else
                {
                    assert( inputType.isF64() );
                }
                result = rewriter.create<LLVM::CallOp>( loc, toyPrintF64(), ValueRange{ input } );
            }
            else if ( inputType.isa<mlir::LLVM::LLVMPointerType>() )
            {
                // Find AllocaOp for size and element type
                int64_t numElems = 0;
                if ( auto loadOp = input.getDefiningOp<toy::LoadOp>() )
                {
                    auto varName = loadOp.getName();
                    auto allocaOp = symbolToAlloca[varName];
                    assert( allocaOp );

                    // Validate element type is i8
                    auto elemType = allocaOp.getElemType();
                    assert( elemType.isa<mlir::IntegerType>() &&
                            ( elemType.cast<mlir::IntegerType>().getWidth() == 8 ) );    // must be i8

                    // Get array size
                    if ( auto constOp = allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
                    {
                        auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>( constOp.getValue() );
                        numElems = intAttr.getInt();
                    }
                }
                else if ( auto stringLitOp = input.getDefiningOp<toy::StringLiteralOp>() )
                {
                    auto strAttr = stringLitOp.getValueAttr();
                    llvm::StringRef strValue = strAttr.getValue();
                    numElems = strValue.size();

                    mlir::LLVM::GlobalOp globalOp = lookupGlobalOp( strAttr );
                    input = rewriter.create<LLVM::AddressOfOp>( loc, globalOp );
                }
                else
                {
                    assert( 0 ); // should not get here.
                }
                assert( numElems );

                auto sizeConst = rewriter.create<mlir::LLVM::ConstantOp>( loc, rewriter.getI64Type(),
                                                                          rewriter.getI64IntegerAttr( numElems ) );

                result = rewriter.create<mlir::LLVM::CallOp>( loc, toyPrintString(), ValueRange{ sizeConst, input } );
            }
            else
            {
                assert( 0 );    // Error: unsupported type
            }

            return result;
        }
    };

    // Lower toy.program to an LLVM function.
    class ProgramOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ProgramOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::ProgramOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto programOp = cast<toy::ProgramOp>( op );
            auto loc = programOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.program: " << *op << '\n' << loc << '\n' );

            Block* entryBlock = lState.addEntryBlock( rewriter );

            // Inline the toy.program's region into the function's entry block
            Region& programRegion = programOp.getRegion();
            if ( !programRegion.hasOneBlock() )
            {
                return rewriter.notifyMatchFailure( programOp, "toy.program must have exactly one block" );
            }

            // Move the block's operations (e.g., toy.return) into the entry
            // block
            Block& programBlock = programRegion.front();
            rewriter.inlineRegionBefore( programRegion, entryBlock );
            rewriter.mergeBlocks( &programBlock, entryBlock,
                                  /* argValues */ {} );

            // Erase the original program op
            rewriter.eraseOp( op );
            // LLVM_DEBUG(llvm::dbgs() << "IR after erasing toy.program:\n" << *op->getParentOp() << "\n");

            // Recursively convert the inlined operations (e.g., toy.return)
            return success();
        }
    };

    class DeclareOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        DeclareOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::DeclareOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto declareOp = cast<toy::DeclareOp>( op );
            auto loc = declareOp.getLoc();

            //   toy.declare "x" : i32
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.declare: " << declareOp << '\n' );

            auto varName = declareOp.getName();
            auto elemType = declareOp.getType();
            // int64_t numElements = 1;    // scalar only for now.

            if ( !elemType.isIntOrFloat() )
            {
                return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
            }

            unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();
            // unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;

#if 0    // FIXME: could pack array creation for i1 types.  For now, just use a separate byte for each.
            if ( elemType.isInteger( 1 ) )
            {
                ...
            }
#endif

            auto ptrType = LLVM::LLVMPointerType::get( rewriter.getContext() );
            auto module = op->getParentOfType<ModuleOp>();
            if ( !module )
            {
                return rewriter.notifyMatchFailure( declareOp, "declare op must be inside a module" );
            }

            mlir::DataLayout dataLayout( module );
            unsigned alignment = dataLayout.getTypePreferredAlignment( elemType );

            rewriter.setInsertionPoint( op );

            mlir::Value sizeVal;
            int64_t arraySize = 1;
            if ( declareOp.getSize().has_value() )
            {
                // Array: Use size attribute, no caching in lState
                arraySize = declareOp.getSize().value();
                if ( arraySize <= 0 )
                {
                    return rewriter.notifyMatchFailure( declareOp, "array size must be positive" );
                }
                sizeVal = rewriter.create<mlir::LLVM::ConstantOp>( loc, rewriter.getI64Type(),
                                                                   rewriter.getI64IntegerAttr( arraySize ) );
            }
            else
            {
                sizeVal = lState.getI64one( loc, rewriter );
            }

            auto allocaOp = rewriter.create<LLVM::AllocaOp>( loc, ptrType, elemType, sizeVal, alignment );

            lState.constructVariableDI( varName, elemType, getLocation( loc ), elemSizeInBits, allocaOp, arraySize );

            auto parentOp = op->getParentOp();

            // Erase the declare op
            rewriter.eraseOp( op );

            LLVM_DEBUG( llvm::dbgs() << "IR after lowering toy.declare:\n" << parentOp << '\n' );

            return success();
        }
    };

    class StringLiteralOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        StringLiteralOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( toy::StringLiteralOp::getOperationName(), 1, ctx ), lState( lState_ )
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto stringLiteralOp = cast<toy::StringLiteralOp>( op );
            auto loc = stringLiteralOp.getLoc();

            auto strAttr = stringLiteralOp.getValueAttr();
            auto strValue = strAttr.getValue().str();
            auto strLen = strValue.size();

            auto globalOp = lState.lookupOrInsertGlobalOp( rewriter, strAttr, loc, strLen, strLen );
            if ( !globalOp )
            {
                return rewriter.notifyMatchFailure( op, "Failed to create or lookup string literal global" );
            }

            auto addr = rewriter.create<LLVM::AddressOfOp>( loc, globalOp );

            // Replace the string literal op with the pointer to the global
            rewriter.replaceOp( op, addr.getResult() );
            return success();
        }
    };

    // Lower AssignOp to llvm.store (after type conversions, if required)
    class AssignOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        AssignOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( toy::AssignOp::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto assignOp = cast<toy::AssignOp>( op );
            auto loc = assignOp.getLoc();

            // (ins StrAttr:$name, AnyType:$value);
            // toy.assign "x", %0 : i32
            LLVM_DEBUG( llvm::dbgs() << "Lowering AssignOp: " << *op << '\n' );

            auto name = assignOp.getName();
            auto value = assignOp.getValue();
            auto valType = value.getType();
            auto allocaOp = lState.symbolToAlloca[name];

            // name: i1v
            // value: %true = arith.constant true
            // valType: i1
            LLVM_DEBUG( llvm::dbgs() << "name: " << name << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "value: " << value << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "valType: " << valType << '\n' );
            // allocaOp.dump(); // %1 = llvm.alloca %0 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr

            // extract parameters from the allocaOp so we know what to do here:
            Type elemType = allocaOp.getElemType();
            int64_t numElems = 0;
            if ( auto constOp = allocaOp.getArraySize().getDefiningOp<LLVM::ConstantOp>() )
            {
                auto intAttr = mlir::dyn_cast<IntegerAttr>( constOp.getValue() );
                numElems = intAttr.getInt();
            }

            // LLVM_DEBUG( llvm::dbgs() << "memType: " << memType << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );
            // LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );

            if ( numElems == 1 )
            {
                if ( mlir::isa<mlir::Float64Type>( valType ) )
                {
                    if ( mlir::isa<mlir::IntegerType>( elemType ) )
                    {
                        value = rewriter.create<LLVM::FPToSIOp>( loc, elemType, value );
                    }
                    else if ( mlir::isa<mlir::Float32Type>( elemType ) )
                    {
                        value = rewriter.create<LLVM::FPTruncOp>( loc, elemType, value );
                    }
                }
                else if ( mlir::isa<mlir::Float32Type>( valType ) )
                {
                    if ( mlir::isa<mlir::IntegerType>( elemType ) )
                    {
                        value = rewriter.create<LLVM::FPToSIOp>( loc, elemType, value );
                    }
                    else if ( mlir::isa<mlir::Float64Type>( elemType ) )
                    {
                        value = rewriter.create<LLVM::FPExtOp>( loc, elemType, value );
                    }
                }
                else if ( auto viType = mlir::cast<mlir::IntegerType>( valType ) )
                {
                    auto vwidth = viType.getWidth();

                    if ( mlir::isa<mlir::Float64Type>( elemType ) || mlir::isa<mlir::Float32Type>( elemType ) )
                    {
                        if ( vwidth == 1 )
                        {
                            value = rewriter.create<LLVM::UIToFPOp>( loc, elemType, value );
                        }
                        else
                        {
                            value = rewriter.create<LLVM::SIToFPOp>( loc, elemType, value );
                        }
                    }
                    else
                    {
                        auto miType = mlir::cast<mlir::IntegerType>( elemType );

                        auto mwidth = miType.getWidth();
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

                rewriter.create<LLVM::StoreOp>( loc, value, allocaOp );
            }
            else if ( auto stringLitOp = value.getDefiningOp<toy::StringLiteralOp>() )
            {
                if ( !mlir::isa<mlir::IntegerType>( elemType ) || elemType.getIntOrFloatBitWidth() != 8 )
                {
                    return rewriter.notifyMatchFailure( assignOp, "string assignment requires i8 array" );
                }
                if ( numElems == 0 )
                {
                    return rewriter.notifyMatchFailure( assignOp, "invalid array size" );
                }

                auto strAttr = stringLitOp.getValueAttr();
                llvm::StringRef strValue = strAttr.getValue();
                size_t literalStrLen = strValue.size();
                mlir::LLVM::GlobalOp globalOp = lState.lookupGlobalOp( strAttr );

                auto globalPtr = rewriter.create<LLVM::AddressOfOp>( loc, globalOp );

                auto destPtr = allocaOp.getResult();

                auto copySize = std::min( (int)numElems, (int)literalStrLen );
                auto sizeConst = rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI64Type(),
                                                                    rewriter.getI64IntegerAttr( copySize ) );

                rewriter.create<LLVM::MemcpyOp>( loc, destPtr, globalPtr, sizeConst, rewriter.getBoolAttr( false ) );

                // If target array is larger than string literal, zero out the remaining bytes
                if ( numElems > literalStrLen )
                {
                    // Compute the offset: destPtr + literalStrLen
                    auto offsetConst = rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI64Type(),
                                                                          rewriter.getI64IntegerAttr( literalStrLen ) );
                    auto destPtrOffset = rewriter.create<LLVM::GEPOp>( loc, destPtr.getType(), elemType, destPtr,
                                                                       ValueRange{ offsetConst } );

                    // Compute the number of bytes to zero: numElems - literalStrLen
                    auto remainingSize = rewriter.create<LLVM::ConstantOp>(
                        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr( numElems - literalStrLen ) );

                    // Set remaining bytes to zero
                    auto zeroVal =
                        rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr( 0 ) );
                    rewriter.create<LLVM::MemsetOp>( loc, destPtrOffset, zeroVal, remainingSize,
                                                     rewriter.getBoolAttr( false ) );
                }
            }
            else
            {
                llvm_unreachable( "AssignOp lowering: expect only fixed size floating or integer types." );
            }

            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower LessOp, ... (after type conversions, if required)
    template <class ToyOp, class IOpType, class FOpType, mlir::LLVM::ICmpPredicate ICmpPredS,
              mlir::LLVM::ICmpPredicate ICmpPredU, mlir::LLVM::FCmpPredicate FCmpPred>
    class ComparisonOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ComparisonOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( ToyOp::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto compareOp = cast<ToyOp>( op );
            auto loc = compareOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering ComparisonOp: " << *op << '\n' );

            auto lhs = compareOp.getLhs();
            auto rhs = compareOp.getRhs();

            auto lhsi = mlir::dyn_cast<IntegerType>( lhs.getType() );
            auto rhsi = mlir::dyn_cast<IntegerType>( rhs.getType() );
            auto lhsf = mlir::dyn_cast<FloatType>( lhs.getType() );
            auto rhsf = mlir::dyn_cast<FloatType>( rhs.getType() );

            if ( lhsi && rhsi )
            {
                auto lwidth = lhsi.getWidth();
                auto rwidth = rhsi.getWidth();
                auto pred = ICmpPredS;

                if ( rwidth > lwidth )
                {
                    if ( lwidth == 1 )
                    {
                        lhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, rhsi, lhs );
                    }
                    else
                    {
                        lhs = rewriter.create<mlir::LLVM::SExtOp>( loc, rhsi, lhs );
                    }
                }
                else if ( rwidth < lwidth )
                {
                    if ( rwidth == 1 )
                    {
                        rhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, lhsi, rhs );
                    }
                    else
                    {
                        rhs = rewriter.create<mlir::LLVM::SExtOp>( loc, lhsi, rhs );
                    }
                }
                else if ( ( rwidth == lwidth ) && ( rwidth == 1 ) )
                {
                    pred = ICmpPredU;
                }

                auto cmp = rewriter.create<IOpType>( loc, pred, lhs, rhs );
                rewriter.replaceOp( op, cmp.getResult() );
            }
            else if ( lhsf && rhsf )
            {
                auto lwidth = lhsf.getWidth();
                auto rwidth = rhsf.getWidth();

                if ( lwidth < rwidth )
                {
                    lhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, rhsf, lhs );
                }
                else if ( rwidth < lwidth )
                {
                    rhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, lhsf, rhs );
                }

                auto cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
                rewriter.replaceOp( op, cmp.getResult() );
            }
            else
            {
                // convert integer type to float
                if ( lhsi && rhsf )
                {
                    if ( lhsi.getWidth() == 1 )
                    {
                        lhs = rewriter.create<mlir::arith::UIToFPOp>( loc, rhsf, lhs );
                    }
                    else
                    {
                        lhs = rewriter.create<mlir::arith::SIToFPOp>( loc, rhsf, lhs );
                    }
                }
                else if ( rhsi && lhsf )
                {
                    if ( rhsi.getWidth() == 1 )
                    {
                        rhs = rewriter.create<mlir::arith::UIToFPOp>( loc, lhsf, rhs );
                    }
                    else
                    {
                        rhs = rewriter.create<mlir::arith::SIToFPOp>( loc, lhsf, rhs );
                    }
                }
                else
                {
                    return rewriter.notifyMatchFailure( op, "Unsupported type combination" );
                }

                auto cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
                rewriter.replaceOp( op, cmp.getResult() );
            }

            return success();
        }
    };

    using LessOpLowering =
        ComparisonOpLowering<toy::LessOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, LLVM::ICmpPredicate::slt,
                             LLVM::ICmpPredicate::ult, mlir::LLVM::FCmpPredicate::olt>;

    using LessEqualOpLowering =
        ComparisonOpLowering<toy::LessEqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, LLVM::ICmpPredicate::sle,
                             LLVM::ICmpPredicate::ule, mlir::LLVM::FCmpPredicate::ole>;

    using EqualOpLowering =
        ComparisonOpLowering<toy::EqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::eq,
                             mlir::LLVM::ICmpPredicate::eq, mlir::LLVM::FCmpPredicate::oeq>;

    using NotEqualOpLowering =
        ComparisonOpLowering<toy::NotEqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::ne,
                             mlir::LLVM::ICmpPredicate::ne, mlir::LLVM::FCmpPredicate::one>;

    class LoadOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        LoadOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::LoadOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto loadOp = cast<toy::LoadOp>( op );
            auto loc = loadOp.getLoc();

            // %0 = toy.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.load: " << *op << '\n' );

            auto name = loadOp.getName();
            auto allocaOp = lState.symbolToAlloca[name];

            LLVM_DEBUG( llvm::dbgs() << "name: " << name << '\n' );

            Type elemType = allocaOp.getElemType();

            if ( loadOp.getResult().getType().isa<mlir::LLVM::LLVMPointerType>() )
            {
                // Return the allocated pointer
                LLVM_DEBUG( llvm::dbgs() << "Loading array address: " << allocaOp.getResult() << '\n' );
                rewriter.replaceOp( op, allocaOp.getResult() );
            }
            else
            {
                // Scalar load
                auto load = rewriter.create<LLVM::LoadOp>( loc, elemType, allocaOp );
                LLVM_DEBUG( llvm::dbgs() << "new load op: " << load << '\n' );
                rewriter.replaceOp( op, load.getResult() );
            }

            return success();
        }
    };

    // Lower toy.return to nothing (erase).
    class ExitOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ExitOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::ExitOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.return: " << *op << '\n' );

            mlir::Location loc = op->getLoc();

            if ( op->getNumOperands() == 0 )
            {
                // RETURN; or default -> return 0
                auto zero = lState.getI32zero( loc, rewriter );
                rewriter.create<LLVM::ReturnOp>( loc, zero );
            }
            else if ( op->getNumOperands() == 1 )
            {
                toy::ExitOp returnOp = cast<toy::ExitOp>( op );

                // RETURN 3; or RETURN x;
                auto operand = returnOp.getRc()[0];

                LLVM_DEBUG( {
                    llvm::dbgs() << "Operand before type conversions:\n";
                    operand.dump();
                } );

                if ( mlir::isa<mlir::Float32Type>( operand.getType() ) ||
                     mlir::isa<mlir::Float64Type>( operand.getType() ) )
                {
                    operand = rewriter.create<LLVM::FPToSIOp>( loc, rewriter.getI32Type(), operand );
                }

                auto intType = mlir::cast<mlir::IntegerType>( operand.getType() );
                auto width = intType.getWidth();
                if ( width > 32 )
                {
                    operand = rewriter.create<mlir::LLVM::TruncOp>( loc, rewriter.getI32Type(), operand );
                }
                else if ( width != 32 )
                {
                    // SExtOp for sign extend.
                    operand = rewriter.create<mlir::LLVM::ZExtOp>( loc, rewriter.getI32Type(), operand );
                }

                LLVM_DEBUG( {
                    llvm::dbgs() << "Final return operand:\n";
                    operand.dump();
                } );

                rewriter.create<LLVM::ReturnOp>( loc, operand );
            }
            else
            {
                llvm_unreachable( "toy.return expects 0 or 1 operands" );
            }

            rewriter.eraseOp( op );
            return success();
        }
    };

#if 0    // Now unused.
    template <class toyOpType>
    class LowerByDeletion : public ConversionPattern
    {
       public:
        LowerByDeletion( MLIRContext* context ) : ConversionPattern( toyOpType::getOperationName(), 1, context )
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs() << "Lowering (by erase): " << *op << '\n' );
            rewriter.eraseOp( op );
            return success();
        }
    };
#endif

    // Lower toy.print to a call to __toy_print.
    class PrintOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        PrintOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::PrintOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto printOp = cast<toy::PrintOp>( op );
            auto loc = printOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.print: " << *op << '\n' );

            mlir::Value input = printOp.getInput();
            LLVM_DEBUG( llvm::dbgs() << "input: " << input << '\n' );

            mlir::LLVM::CallOp result = lState.createPrintCall( rewriter, loc, input );

            rewriter.replaceOp( op, result );
            return success();
        }
    };

    // Lower toy.negate to LLVM arithmetic.
    class NegOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        NegOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::NegOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto unaryOp = cast<toy::NegOp>( op );
            auto loc = unaryOp.getLoc();
            auto operand = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.negate: " << *op << '\n' );

            mlir::Value result = operand;

            auto zero = lState.getF64zero( loc, rewriter );
            result = rewriter.create<LLVM::FSubOp>( loc, zero, result );

            rewriter.replaceOp( op, result );
            return success();
        }
    };

    // Lower toy.binary to LLVM arithmetic.
    template <class ToyBinaryOpType, class llvmIOpType, class llvmFOpType, bool allowFloat>
    class BinaryOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        BinaryOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( ToyBinaryOpType::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto binaryOp = cast<ToyBinaryOpType>( op );
            auto loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.binary: " << *op << '\n' );

            auto resultType = binaryOp.getResult().getType();

            Value lhs = operands[0];
            Value rhs = operands[1];
            if ( resultType.isIntOrIndex() )
            {
                auto rwidth = resultType.getIntOrFloatBitWidth();

                if ( auto lhsi = mlir::dyn_cast<IntegerType>( lhs.getType() ) )
                {
                    auto width = lhsi.getWidth();

                    if ( rwidth > width )
                    {
                        lhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, resultType, lhs );
                    }
                    else if ( rwidth < width )
                    {
                        lhs = rewriter.create<mlir::LLVM::TruncOp>( loc, resultType, lhs );
                    }
                }
                else if ( lhs.getType().isF32() || lhs.getType().isF64() )
                {
                    if ( allowFloat )
                    {
                        lhs = rewriter.create<LLVM::FPToSIOp>( loc, resultType, lhs );
                    }
                    else
                    {
                        llvm_unreachable( "float types unsupported for integer binary operation" );
                    }
                }

                if ( auto rhsi = mlir::dyn_cast<IntegerType>( rhs.getType() ) )
                {
                    auto width = rhsi.getWidth();

                    if ( rwidth > width )
                    {
                        rhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, resultType, rhs );
                    }
                    else if ( rwidth < width )
                    {
                        rhs = rewriter.create<mlir::LLVM::TruncOp>( loc, resultType, rhs );
                    }
                }
                else if ( rhs.getType().isF32() || rhs.getType().isF64() )
                {
                    if ( allowFloat )
                    {
                        rhs = rewriter.create<LLVM::FPToSIOp>( loc, resultType, rhs );
                    }
                    else
                    {
                        llvm_unreachable( "float types unsupported for integer binary operation" );
                    }
                }

                auto result = rewriter.create<llvmIOpType>( loc, lhs, rhs );
                rewriter.replaceOp( op, result );
            }
            else if ( allowFloat )
            {
                // Floating-point addition: ensure both operands are f64.
                if ( auto lhsi = mlir::dyn_cast<IntegerType>( lhs.getType() ) )
                {
                    auto width = lhsi.getWidth();

                    if ( width == 1 )
                    {
                        lhs = rewriter.create<LLVM::UIToFPOp>( loc, resultType, lhs );
                    }
                    else
                    {
                        lhs = rewriter.create<LLVM::SIToFPOp>( loc, resultType, lhs );
                    }
                }
                if ( auto rhsi = mlir::dyn_cast<IntegerType>( rhs.getType() ) )
                {
                    auto width = rhsi.getWidth();

                    if ( width == 1 )
                    {
                        rhs = rewriter.create<LLVM::UIToFPOp>( loc, resultType, rhs );
                    }
                    else
                    {
                        rhs = rewriter.create<LLVM::SIToFPOp>( loc, resultType, rhs );
                    }
                }
                auto result = rewriter.create<llvmFOpType>( loc, lhs, rhs );
                rewriter.replaceOp( op, result );
            }
            else
            {
                llvm_unreachable( "float types unsupported for integer binary operation" );
            }

            return success();
        }
    };

    using AddOpLowering = BinaryOpLowering<toy::AddOp, LLVM::AddOp, LLVM::FAddOp, true>;
    using SubOpLowering = BinaryOpLowering<toy::SubOp, LLVM::SubOp, LLVM::FSubOp, true>;
    using MulOpLowering = BinaryOpLowering<toy::MulOp, LLVM::MulOp, LLVM::FMulOp, true>;
    using DivOpLowering = BinaryOpLowering<toy::DivOp, LLVM::SDivOp, LLVM::FDivOp, true>;

    // LLVM::FAddOp is a dummy operation here, knowing that it will not ever be used:
    using XorOpLowering = BinaryOpLowering<toy::XorOp, mlir::LLVM::XOrOp, LLVM::FAddOp, false>;
    using AndOpLowering = BinaryOpLowering<toy::AndOp, mlir::LLVM::AndOp, LLVM::FAddOp, false>;
    using OrOpLowering = BinaryOpLowering<toy::OrOp, mlir::LLVM::OrOp, LLVM::FAddOp, false>;

    // Lower arith.constant to LLVM constant.
    class ConstantOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ConstantOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( arith::ConstantOp::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto constantOp = cast<arith::ConstantOp>( op );
            auto loc = constantOp.getLoc();
            auto valueAttr = constantOp.getValue();

            LLVM_DEBUG( llvm::dbgs() << "Lowering arith.constant: " << *op << '\n' );

            if ( auto fAttr = dyn_cast<FloatAttr>( valueAttr ) )
            {
                auto f64Type = rewriter.getF64Type();    // Returns Float64Type for f64
                auto value = rewriter.create<LLVM::ConstantOp>( loc, f64Type, fAttr );
                rewriter.replaceOp( op, value );
                return success();
            }
            else if ( auto intAttr = dyn_cast<IntegerAttr>( valueAttr ) )
            {
                auto i64Type = IntegerType::get( rewriter.getContext(), 64 );
                auto value = rewriter.create<LLVM::ConstantOp>( loc, i64Type, intAttr );
                rewriter.replaceOp( op, value );
                return success();
            }

            return failure();
        }
    };

    class ToyToLLVMLoweringPass : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( ToyToLLVMLoweringPass )

        ToyToLLVMLoweringPass( toy::driverState* pst_ ) : pDriverState{ pst_ }
        {
        }

        void getDependentDialects( DialectRegistry& registry ) const override
        {
            registry.insert<LLVM::LLVMDialect, arith::ArithDialect>();
        }

        void runOnOperation() override
        {
            ModuleOp module = getOperation();    // really an operation wrapped ModuleOp, not the same as, say,
                                                 // mlir::ModuleOp::create( loc );

            LLVM_DEBUG( {
                llvm::dbgs() << "Starting ToyToLLVMLoweringPass on:\n";
                module->dump();
            } );

            loweringContext lState( module, *pDriverState );

            lState.setModuleAttrs();

            auto ctx = lState.getBuilder().getContext();
            lState.createMain();

            // Initialize the type converter
            LLVMTypeConverter typeConverter( ctx );

            // Conversion target: only LLVM dialect is legal
            ConversionTarget target1( getContext() );
            target1.addLegalDialect<LLVM::LLVMDialect>();
            target1.addIllegalOp<arith::ConstantOp>();
            target1.addIllegalOp<toy::DeclareOp, toy::AssignOp, toy::PrintOp, toy::AddOp, toy::SubOp, toy::MulOp,
                                 toy::DivOp, toy::NegOp, toy::ExitOp>();
            target1.addLegalOp<mlir::ModuleOp>();
            target1.addIllegalDialect<toy::ToyDialect>();

            // Patterns for toy dialect and standard ops
            RewritePatternSet patterns1( ctx );

            // The operator ordering here doesn't matter, as there appears to be a graph walk to find all the operator
            // nodes, and the order is based on that walk (i.e.: ProgramOpLowering happens first.)
            patterns1.insert<DeclareOpLowering>( lState, ctx );
            patterns1.insert<LoadOpLowering>( lState, ctx );
            patterns1.insert<AddOpLowering>( lState, ctx );
            patterns1.insert<SubOpLowering>( lState, ctx );
            patterns1.insert<MulOpLowering>( lState, ctx );
            patterns1.insert<DivOpLowering>( lState, ctx );
            patterns1.insert<NegOpLowering>( lState, ctx );
            patterns1.insert<LessOpLowering>( lState, ctx );
            patterns1.insert<EqualOpLowering>( lState, ctx );
            patterns1.insert<NotEqualOpLowering>( lState, ctx );
            patterns1.insert<XorOpLowering>( lState, ctx );
            patterns1.insert<AndOpLowering>( lState, ctx );
            patterns1.insert<OrOpLowering>( lState, ctx );
            patterns1.insert<LessEqualOpLowering>( lState, ctx );
            patterns1.insert<PrintOpLowering>( lState, ctx );
            patterns1.insert<ConstantOpLowering>( lState, ctx );
            patterns1.insert<AssignOpLowering>( lState, ctx );
            patterns1.insert<StringLiteralOpLowering>( lState, ctx );
            patterns1.insert<ExitOpLowering>( lState, ctx );
            patterns1.insert<ProgramOpLowering>( lState, ctx );

            arith::populateArithToLLVMConversionPatterns( typeConverter, patterns1 );

            if ( failed( applyFullConversion( module, target1, std::move( patterns1 ) ) ) )
            {
                LLVM_DEBUG( llvm::dbgs() << "Conversion failed\n" );
                signalPassFailure();
                return;
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After ToyToLLVMLoweringPass:\n";
                for ( Operation& op : module->getRegion( 0 ).front() )
                {
                    op.dump();
                }
            } );
        }

       private:
        toy::driverState* pDriverState;
    };
}    // namespace toy

namespace mlir
{
    // Parameterless version for TableGen
    std::unique_ptr<Pass> createToyToLLVMLoweringPass()
    {
        return createToyToLLVMLoweringPass( nullptr );    // Default to no optimization
    }

    // Parameterized version
    std::unique_ptr<Pass> createToyToLLVMLoweringPass( toy::driverState* pst )
    {
        return std::make_unique<toy::ToyToLLVMLoweringPass>( pst );
    }

    // Custom registration with bool parameter
    void registerToyToLLVMLoweringPass( toy::driverState* pst )
    {
        ::mlir::registerPass( [pst]() -> std::unique_ptr<::mlir::Pass>
                              { return mlir::createToyToLLVMLoweringPass( pst ); } );
    }
}    // namespace mlir

// vim: et ts=4 sw=4
