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
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

#include "ToyDialect.hpp"
#include "ToyExceptions.hpp"
#include "constants.hpp"
#include "lowering.hpp"

#define DEBUG_TYPE "toy-lowering"

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

    mlir::func::FuncOp getEnclosingFuncOp( mlir::Operation* op )
    {
        while ( op )
        {
            if ( auto funcOp = dyn_cast<mlir::func::FuncOp>( op ) )
            {
                return funcOp;
            }
            op = op->getParentOp();
        }
        return nullptr;
    }

    class loweringContext
    {
       private:
        // Caching these may not be a good idea, as they are created with a single loc value, but using an existing
        // constant is also allowed, so maybe that's okay?
        mlir::LLVM::ConstantOp pr_zero_I8;
        mlir::LLVM::ConstantOp pr_zero_I16;
        mlir::LLVM::ConstantOp pr_zero_I32;
        mlir::LLVM::ConstantOp pr_zero_I64;
        mlir::LLVM::ConstantOp pr_one_I64;
        mlir::LLVM::ConstantOp pr_zero_F32;
        mlir::LLVM::ConstantOp pr_zero_F64;

        mlir::LLVM::DIFileAttr pr_fileAttr;
        std::unordered_map<std::string, mlir::LLVM::DISubprogramAttr> pr_subprogramAttr;
        std::unordered_map<std::string, mlir::LLVM::GlobalOp> pr_stringLiterals;
        mlir::LLVM::LLVMFuncOp pr_printFuncF64;
        mlir::LLVM::LLVMFuncOp pr_printFuncI64;
        mlir::LLVM::LLVMFuncOp pr_printFuncString;

        const toy::driverState& pr_driverState;
        ModuleOp& pr_module;
        OpBuilder pr_builder;
        std::unordered_map<std::string, mlir::Operation*> pr_symbolToAlloca;

        mlir::LLVM::DICompileUnitAttr pr_compileUnitAttr;
        mlir::LLVM::DITypeAttr pr_diI8;
        mlir::LLVM::DITypeAttr pr_diI16;
        mlir::LLVM::DITypeAttr pr_diI32;
        mlir::LLVM::DITypeAttr pr_diI64;
        mlir::LLVM::DITypeAttr pr_diF32;
        mlir::LLVM::DITypeAttr pr_diF64;
        mlir::LLVM::DITypeAttr pr_diVOID;
        mlir::LLVM::DITypeAttr pr_diUNKNOWN;

       public:
        mlir::IntegerType tyI1;
        mlir::IntegerType tyI8;
        mlir::IntegerType tyI16;
        mlir::IntegerType tyI32;
        mlir::IntegerType tyI64;
        mlir::FloatType tyF32;
        mlir::FloatType tyF64;
        mlir::LLVM::LLVMPointerType tyPtr;
        mlir::LLVM::LLVMVoidType tyVoid;
        LLVMTypeConverter typeConverter;

        loweringContext( ModuleOp& module, const toy::driverState& driverState )
            : pr_driverState{ driverState },
              pr_module{ module },
              pr_builder{ module.getRegion() },
              typeConverter{ pr_builder.getContext() }
        {
            tyI1 = pr_builder.getI1Type();
            tyI8 = pr_builder.getI8Type();
            tyI16 = pr_builder.getI16Type();
            tyI32 = pr_builder.getI32Type();
            tyI64 = pr_builder.getI64Type();

            tyF32 = pr_builder.getF32Type();
            tyF64 = pr_builder.getF64Type();

            auto context = pr_builder.getContext();
            tyPtr = LLVM::LLVMPointerType::get( context );

            tyVoid = LLVM::LLVMVoidType::get( context );
        }

        unsigned preferredTypeAlignment( Operation* op, mlir::Type elemType )
        {
            auto module = op->getParentOfType<ModuleOp>();
            assert( module );
            mlir::DataLayout dataLayout( module );
            unsigned alignment = dataLayout.getTypePreferredAlignment( elemType );

            return alignment;
        }

        // Note for future: c++-14 now allows auto-return for a simple function like this.
        mlir::MLIRContext* getContext()
        {
            return pr_builder.getContext();
        }

        bool isTypeFloat( mlir::Type ty ) const
        {
            return ( ( ty == tyF32 ) || ( ty == tyF64 ) );
        }

        mlir::LLVM::ConstantOp getI8zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_I8 )
            {
                pr_zero_I8 = rewriter.create<LLVM::ConstantOp>( loc, tyI8, rewriter.getI8IntegerAttr( 0 ) );
            }

            return pr_zero_I8;
        }

        mlir::LLVM::ConstantOp getI16zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_I16 )
            {
                pr_zero_I16 = rewriter.create<LLVM::ConstantOp>( loc, tyI16, rewriter.getI16IntegerAttr( 0 ) );
            }

            return pr_zero_I16;
        }

        mlir::LLVM::ConstantOp getI32zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_I32 )
            {
                pr_zero_I32 = rewriter.create<LLVM::ConstantOp>( loc, tyI32, rewriter.getI32IntegerAttr( 0 ) );
            }

            return pr_zero_I32;
        }

        mlir::LLVM::ConstantOp getI64zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_I64 )
            {
                pr_zero_I64 = rewriter.create<LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( 0 ) );
            }

            return pr_zero_I64;
        }

        /// Returns a cached zero constant for the given integer width (i8, i16, i32, i64).
        /// Throws an exception for unsupported widths.
        mlir::LLVM::ConstantOp getIzero( mlir::Location loc, ConversionPatternRewriter& rewriter, unsigned width )
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

            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "Unexpected integer size {}", width ) );
        }

        mlir::LLVM::ConstantOp getI64one( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_one_I64 )
            {
                pr_one_I64 = rewriter.create<LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( 1 ) );
            }

            return pr_one_I64;
        }

        mlir::LLVM::ConstantOp getF32zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_F32 )
            {
                pr_zero_F32 = rewriter.create<LLVM::ConstantOp>( loc, tyF32, rewriter.getF32FloatAttr( 0 ) );
            }

            return pr_zero_F32;
        }

        /// Returns a cached zero constant for the given float width (f32, f64).
        /// Throws an exception for unsupported widths.
        mlir::LLVM::ConstantOp getF64zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !pr_zero_F64 )
            {
                pr_zero_F64 = rewriter.create<LLVM::ConstantOp>( loc, tyF64, rewriter.getF64FloatAttr( 0 ) );
            }

            return pr_zero_F64;
        }

        mlir::LLVM::ConstantOp getFzero( mlir::Location loc, ConversionPatternRewriter& rewriter, unsigned width )
        {
            switch ( width )
            {
                case 32:
                    return getF32zero( loc, rewriter );
                case 64:
                    return getF64zero( loc, rewriter );
            }

            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "Unexpected float size {}", width ) );
        }

        class useModuleInsertionPoint
        {
            OpBuilder& builder;
            mlir::OpBuilder::InsertPoint oldIP;

           public:
            useModuleInsertionPoint( ModuleOp& module, OpBuilder& builder_ )
                : builder{ builder_ }, oldIP{ builder.saveInsertionPoint() }
            {
                builder.setInsertionPointToStart( module.getBody() );
            }

            ~useModuleInsertionPoint()
            {
                builder.restoreInsertionPoint( oldIP );
            }
        };

        mlir::LLVM::LLVMFuncOp toyPrintF64()
        {
            if ( !pr_printFuncF64 )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                auto pr_printFuncF64Type = LLVM::LLVMFunctionType::get( tyVoid, { tyF64 }, false );
                pr_printFuncF64 = pr_builder.create<LLVM::LLVMFuncOp>( pr_module.getLoc(), "__toy_print_f64",
                                                                       pr_printFuncF64Type, LLVM::Linkage::External );
            }

            return pr_printFuncF64;
        }

        mlir::LLVM::LLVMFuncOp toyPrintI64()
        {
            if ( !pr_printFuncI64 )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                auto printFuncI64Type = LLVM::LLVMFunctionType::get( tyVoid, { tyI64 }, false );
                pr_printFuncI64 = pr_builder.create<LLVM::LLVMFuncOp>( pr_module.getLoc(), "__toy_print_i64",
                                                                       printFuncI64Type, LLVM::Linkage::External );
            }

            return pr_printFuncI64;
        }

        mlir::LLVM::LLVMFuncOp toyPrintString()
        {
            if ( !pr_printFuncString )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                auto printFuncStringType = LLVM::LLVMFunctionType::get( tyVoid, { tyI64, tyPtr }, false );
                pr_printFuncString = pr_builder.create<LLVM::LLVMFuncOp>(
                    pr_module.getLoc(), "__toy_print_string", printFuncStringType, LLVM::Linkage::External );
            }

            return pr_printFuncString;
        }

        void createDICompileUnit()
        {
            if ( pr_driverState.wantDebug )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                auto context = pr_builder.getContext();


                pr_diVOID = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                              pr_builder.getStringAttr( "void" ), 0, 0 );

                pr_diF32 = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                             pr_builder.getStringAttr( "float" ), 32,
                                                             llvm::dwarf::DW_ATE_float );

                pr_diF64 = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                             pr_builder.getStringAttr( "double" ), 64,
                                                             llvm::dwarf::DW_ATE_float );

                pr_diUNKNOWN = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                                 pr_builder.getStringAttr( "unknown" ), 0, 0 );


                pr_diI8 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                            pr_builder.getStringAttr( "char" ), 8,
                                                            (unsigned)llvm::dwarf::DW_ATE_signed );

                pr_diI16 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                             pr_builder.getStringAttr( "short" ), 16,
                                                             (unsigned)llvm::dwarf::DW_ATE_signed );

                pr_diI32 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                             pr_builder.getStringAttr( "int" ), 32,
                                                             (unsigned)llvm::dwarf::DW_ATE_signed );

                pr_diI64 = mlir::LLVM::DIBasicTypeAttr::get( context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                             pr_builder.getStringAttr( "long" ), 64,
                                                             (unsigned)llvm::dwarf::DW_ATE_signed );

                // Construct pr_module level DI state:
                pr_fileAttr = mlir::LLVM::DIFileAttr::get( context, pr_driverState.filename, "." );
                auto distinctAttr = mlir::DistinctAttr::create( pr_builder.getUnitAttr() );
                pr_compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
                    context, distinctAttr, llvm::dwarf::DW_LANG_C, pr_fileAttr,
                    pr_builder.getStringAttr( COMPILER_NAME ), false, mlir::LLVM::DIEmissionKind::Full,
                    mlir::LLVM::DINameTableKind::Default );
            }

            // Set data_layout,ident,target_triple:
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

        mlir::LLVM::DITypeAttr getDIType( mlir::Type type )
        {
            if ( !type )
            {
                return pr_diVOID;
            }
            else if ( type.isF32() )
            {
                return pr_diF32;
            }
            else if ( type.isF64() )
            {
                return pr_diF64;
            }
            else if ( type.isInteger( 8 ) || type.isInteger( 1 ) )
            {
                return pr_diI8;
            }
            else if ( type.isInteger( 16 ) )
            {
                return pr_diI16;
            }
            else if ( type.isInteger( 32 ) )
            {
                return pr_diI32;
            }
            else if ( type.isInteger( 64 ) )
            {
                return pr_diI64;
            }
            else
            {
                return pr_diUNKNOWN;
            }
        }

        mlir::LLVM::DISubroutineTypeAttr createDISubroutineType( mlir::func::FuncOp funcOp )
        {
            auto funcType = funcOp.getFunctionType();

            mlir::SmallVector<mlir::LLVM::DITypeAttr> paramTypes;

            auto returnType = getDIType( funcType.getResults().empty() ? mlir::Type() : funcType.getResults()[0] );
            paramTypes.push_back( returnType );

            for ( auto argType : funcType.getInputs() )
            {
                paramTypes.push_back( getDIType( argType ) );
            }

            auto context = pr_builder.getContext();

            return mlir::LLVM::DISubroutineTypeAttr::get( context, llvm::DINode::FlagZero, paramTypes );
        }

        void createFuncDebug( mlir::func::FuncOp funcOp )
        {
            if ( pr_driverState.wantDebug )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                auto context = pr_builder.getContext();
                auto funcName = funcOp.getSymName().str();

                auto subprogramType = createDISubroutineType( funcOp );

                auto sub = mlir::LLVM::DISubprogramAttr::get(
                    context, mlir::DistinctAttr::create( pr_builder.getUnitAttr() ), pr_compileUnitAttr, pr_fileAttr,
                    pr_builder.getStringAttr( funcName ), pr_builder.getStringAttr( funcName ), pr_fileAttr, 1, 1,
                    mlir::LLVM::DISubprogramFlags::Definition, subprogramType, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{},
                    llvm::ArrayRef<mlir::LLVM::DINodeAttr>{} );

                funcOp->setAttr( "llvm.debug.subprogram", sub );

                // This is the key to ensure that translateModuleToLLVMIR does not strip the location info (instead
                // converts loc's into !dbg's)
                funcOp->setLoc( pr_builder.getFusedLoc( { pr_module.getLoc() }, sub ) );

                pr_subprogramAttr[funcName] = sub;
            }
        }

        std::string lookupFuncNameForOp( mlir::Operation* op )
        {
            mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

            return funcOp.getSymName().str();
        }

        mlir::LLVM::AllocaOp lookupLocalSymbolReference( mlir::Operation* op, const std::string& varName )
        {
            mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

            LLVM_DEBUG( {
                llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
                funcOp->dump();
            } );

            auto funcName = funcOp->getName().getStringRef().str();

            std::string funcNameAndVarName = funcName + "::" + varName;

            auto alloca = pr_symbolToAlloca[funcNameAndVarName];
            return mlir::dyn_cast<mlir::LLVM::AllocaOp>( alloca );
        }

        void createLocalSymbolReference( mlir::LLVM::AllocaOp allocaOp, const std::string& varName )
        {
            auto funcOp = getEnclosingFuncOp( allocaOp );
            auto funcName = funcOp->getName().getStringRef().str();

            std::string funcNameAndVarName = funcName + "::" + varName;

            pr_symbolToAlloca[funcNameAndVarName] = allocaOp;
        }

        void constructVariableDI( llvm::StringRef varName, mlir::Type& elemType, mlir::FileLineColLoc loc,
                                  unsigned elemSizeInBits, mlir::LLVM::AllocaOp& allocaOp, int64_t arraySize = 1 )
        {
            if ( pr_driverState.wantDebug )
            {
                auto context = pr_builder.getContext();

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

                std::string funcName = lookupFuncNameForOp( allocaOp );
                auto sub = pr_subprogramAttr[funcName];
                assert( sub );

                unsigned totalSizeInBits = elemStorageSizeInBits * arraySize;
                if ( arraySize > 1 )
                {
                    // Create base type for array elements
                    auto baseType = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                                      pr_builder.getStringAttr( typeName ),
                                                                      elemStorageSizeInBits, dwType );

                    // Create subrange for array (count = arraySize, lowerBound = 0)
                    auto countAttr = mlir::IntegerAttr::get( tyI64, arraySize );
                    auto lowerBoundAttr = mlir::IntegerAttr::get( tyI64, 0 );
                    auto subrange = mlir::LLVM::DISubrangeAttr::get( context, countAttr, lowerBoundAttr,
                                                                     /*upperBound=*/nullptr, /*stride=*/nullptr );

                    // Create array type
                    auto alignInBits = elemStorageSizeInBits;    // Alignment matches element size
                    diType = mlir::LLVM::DICompositeTypeAttr::get(
                        context, llvm::dwarf::DW_TAG_array_type, pr_builder.getStringAttr( "" ), pr_fileAttr,
                        /*line=*/0, sub, baseType, mlir::LLVM::DIFlags::Zero, totalSizeInBits, alignInBits,
                        llvm::ArrayRef<mlir::LLVM::DINodeAttr>{ subrange },
                        /*dataLocation=*/nullptr, /*rank=*/nullptr, /*allocated=*/nullptr, /*associated=*/nullptr );
                }
                else
                {
                    // Scalar type
                    diType = mlir::LLVM::DIBasicTypeAttr::get( context, llvm::dwarf::DW_TAG_base_type,
                                                               pr_builder.getStringAttr( typeName ),
                                                               elemStorageSizeInBits, dwType );
                }

                diVar = mlir::LLVM::DILocalVariableAttr::get(
                    context, sub, pr_builder.getStringAttr( varName ), pr_fileAttr, loc.getLine(),
                    /*argNo=*/0, totalSizeInBits, diType, mlir::LLVM::DIFlags::Zero );

                pr_builder.setInsertionPointAfter( allocaOp );
                pr_builder.create<mlir::LLVM::DbgDeclareOp>( loc, allocaOp, diVar );
            }

            createLocalSymbolReference( allocaOp, varName.str() );
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
                                                     mlir::Location loc, size_t strLen )
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

                auto arrayType = mlir::LLVM::LLVMArrayType::get( tyI8, strLen );

                SmallVector<char> stringData( stringLit.begin(), stringLit.end() );
                auto denseAttr = DenseElementsAttr::get(
                    RankedTensorType::get( { static_cast<int64_t>( strLen ) }, tyI8 ), ArrayRef<char>( stringData ) );

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
                    input = rewriter.create<mlir::LLVM::ZExtOp>( loc, tyI64, input );
                }
                else if ( width < 64 )
                {
                    input = rewriter.create<mlir::LLVM::SExtOp>( loc, tyI64, input );
                }

                result = rewriter.create<LLVM::CallOp>( loc, toyPrintI64(), ValueRange{ input } );
            }
            else if ( auto inputf = mlir::dyn_cast<FloatType>( inputType ) )
            {
                if ( inputType == tyF32 )
                {
                    input = rewriter.create<LLVM::FPExtOp>( loc, tyF64, input );
                }
                else
                {
                    assert( inputType == tyF64 );
                }
                result = rewriter.create<LLVM::CallOp>( loc, toyPrintF64(), ValueRange{ input } );
            }
            else if ( inputType == tyPtr )
            {
                // Find AllocaOp for size and element type
                int64_t numElems = 0;
                if ( auto loadOp = input.getDefiningOp<toy::LoadOp>() )
                {
                    auto varNameAttr = loadOp.getVarName();
                    assert( varNameAttr );

                    // Get string (e.g., "x")
                    auto varName = varNameAttr.getLeafReference().str();
                    LLVM_DEBUG( { llvm::dbgs() << "LoadOp variable name: " << varName << "\n"; } );

                    auto allocaOp = lookupLocalSymbolReference( loadOp, varName );

                    // Validate element type is i8
                    auto elemType = allocaOp.getElemType();
                    assert( elemType == tyI8 );

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
                    assert( 0 );    // should not get here.
                }
                assert( numElems );

                auto sizeConst =
                    rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( numElems ) );

                result = rewriter.create<mlir::LLVM::CallOp>( loc, toyPrintString(), ValueRange{ sizeConst, input } );
            }
            else
            {
                assert( 0 );    // Error: unsupported type
            }

            return result;
        }

        /// Emit debug info for parameter
        void generateParamDI( mlir::FileLineColLoc loc, ConversionPatternRewriter& rewriter, const std::string& varName,
                              mlir::BlockArgument value, mlir::Type elemType, int paramIndex,
                              const std::string& funcName )
        {
            // Create debug type for basic types (e.g., i32, f32)
            auto* context = rewriter.getContext();
            auto diType = getDIType( elemType );

            // Get DISubprogram from pr_subprogramAttr
            auto sub = pr_subprogramAttr[funcName];
            assert( sub );

            unsigned bitWidth = elemType.getIntOrFloatBitWidth();

            // Create debug variable
            auto diVar = mlir::LLVM::DILocalVariableAttr::get( context, sub, rewriter.getStringAttr( varName ),
                                                               pr_fileAttr, loc.getLine(), paramIndex + 1, bitWidth,
                                                               diType, mlir::LLVM::DIFlags::Zero );

            // Emit llvm.dbg.declare
            rewriter.create<mlir::LLVM::DbgDeclareOp>( loc, value, diVar );
        }
    };

    class DeclareOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        DeclareOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::DeclareOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto declareOp = cast<toy::DeclareOp>( op );
            auto loc = declareOp.getLoc();
            auto param = declareOp.isParameter();

            //   toy.declare "x" : i32
            LLVM_DEBUG( llvm::dbgs() << std::format( "Lowering toy.declare: param: {}", param ) << declareOp << '\n' );

            rewriter.setInsertionPoint( op );

            auto varName = declareOp.getName();
            auto elemType = declareOp.getType();

            if ( param )
            {
                auto paramNumberAttr = declareOp.getParamNumberAttr();
                if ( !paramNumberAttr )
                {
                    return rewriter.notifyMatchFailure( op, "Parameter missing param_number attribute" );
                }
                int64_t paramIndex = paramNumberAttr.getInt();
                auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
                if ( paramIndex >= funcOp.getNumArguments() )
                {
                    return rewriter.notifyMatchFailure( op, "Parameter index out of bounds" );
                }
                auto value = funcOp.getArgument( paramIndex );

                auto funcName = funcOp.getSymName().str();

                lState.generateParamDI( getLocation( loc ), rewriter, varName.str(), value, elemType, paramIndex, funcName );
            }
            else
            {
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

                auto allocaOp = rewriter.create<LLVM::AllocaOp>( loc, lState.tyPtr, elemType, sizeVal, alignment );
                lState.constructVariableDI( varName, elemType, getLocation( loc ), elemSizeInBits, allocaOp,
                                            arraySize );
            }

            rewriter.eraseOp( op );

            return success();
        }
    };

    class StringLiteralOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        StringLiteralOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::StringLiteralOp::getOperationName(), benefit, context ), lState( lState_ )
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

            auto globalOp = lState.lookupOrInsertGlobalOp( rewriter, strAttr, loc, strLen );
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
        AssignOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::AssignOp::getOperationName(), benefit, context ), lState{ lState_ }
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

            auto varNameAttr = assignOp.getVarName();
            assert( varNameAttr );

            // Get string (e.g., "x")
            auto varName = varNameAttr.getLeafReference().str();
            LLVM_DEBUG( { llvm::dbgs() << "LoadOp variable name: " << varName << "\n"; } );

            auto allocaOp = lState.lookupLocalSymbolReference( assignOp, varName );

            auto value = assignOp.getValue();
            auto valType = value.getType();

            // varName: i1v
            // value: %true = arith.constant true
            // valType: i1
            LLVM_DEBUG( llvm::dbgs() << "varName: " << varName << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "value: " << value << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "valType: " << valType << '\n' );

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
                if ( valType == lState.tyF64 )
                {
                    if ( mlir::isa<mlir::IntegerType>( elemType ) )
                    {
                        value = rewriter.create<LLVM::FPToSIOp>( loc, elemType, value );
                    }
                    else if ( elemType == lState.tyF32 )
                    {
                        value = rewriter.create<LLVM::FPTruncOp>( loc, elemType, value );
                    }
                }
                else if ( valType == lState.tyF32 )
                {
                    if ( mlir::isa<mlir::IntegerType>( elemType ) )
                    {
                        value = rewriter.create<LLVM::FPToSIOp>( loc, elemType, value );
                    }
                    else if ( elemType == lState.tyF64 )
                    {
                        value = rewriter.create<LLVM::FPExtOp>( loc, elemType, value );
                    }
                }
                else if ( auto viType = mlir::cast<mlir::IntegerType>( valType ) )
                {
                    auto vwidth = viType.getWidth();
                    if ( lState.isTypeFloat( elemType ) )
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
                    else if ( auto miType = mlir::cast<mlir::IntegerType>( elemType ) )
                    {
                        auto mwidth = miType.getWidth();
                        if ( vwidth > mwidth )
                        {
                            value = rewriter.create<LLVM::TruncOp>( loc, elemType, value );
                        }
                        else if ( vwidth < mwidth )
                        {
                            value = rewriter.create<LLVM::ZExtOp>( loc, elemType, value );
                        }
                    }
                }

                unsigned alignment = lState.preferredTypeAlignment( op, elemType );
                rewriter.create<LLVM::StoreOp>( loc, value, allocaOp, alignment );
            }
            else if ( auto stringLitOp = value.getDefiningOp<toy::StringLiteralOp>() )
            {
                if ( elemType != lState.tyI8 )
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
                auto sizeConst =
                    rewriter.create<LLVM::ConstantOp>( loc, lState.tyI64, rewriter.getI64IntegerAttr( copySize ) );

                rewriter.create<LLVM::MemcpyOp>( loc, destPtr, globalPtr, sizeConst, rewriter.getBoolAttr( false ) );

                // If target array is larger than string literal, zero out the remaining bytes
                if ( numElems > (int64_t)literalStrLen )
                {
                    // Compute the offset: destPtr + literalStrLen
                    auto offsetConst = rewriter.create<LLVM::ConstantOp>( loc, lState.tyI64,
                                                                          rewriter.getI64IntegerAttr( literalStrLen ) );
                    auto destPtrOffset = rewriter.create<LLVM::GEPOp>( loc, destPtr.getType(), elemType, destPtr,
                                                                       ValueRange{ offsetConst } );

                    // Compute the number of bytes to zero: numElems - literalStrLen
                    auto remainingSize = rewriter.create<LLVM::ConstantOp>(
                        loc, lState.tyI64, rewriter.getI64IntegerAttr( numElems - literalStrLen ) );

                    // Set remaining bytes to zero
                    rewriter.create<LLVM::MemsetOp>( loc, destPtrOffset, lState.getI8zero( loc, rewriter ),
                                                     remainingSize, rewriter.getBoolAttr( false ) );
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
        ComparisonOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( ToyOp::getOperationName(), benefit, context ), lState{ lState_ }
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

            auto lTyI = mlir::dyn_cast<IntegerType>( lhs.getType() );
            auto rTyI = mlir::dyn_cast<IntegerType>( rhs.getType() );
            auto lTyF = mlir::dyn_cast<FloatType>( lhs.getType() );
            auto rTyF = mlir::dyn_cast<FloatType>( rhs.getType() );

            if ( lTyI && rTyI )
            {
                auto lwidth = lTyI.getWidth();
                auto rwidth = rTyI.getWidth();
                auto pred = ICmpPredS;

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

                auto cmp = rewriter.create<IOpType>( loc, pred, lhs, rhs );
                rewriter.replaceOp( op, cmp.getResult() );
            }
            else if ( lTyF && rTyF )
            {
                auto lwidth = lTyF.getWidth();
                auto rwidth = rTyF.getWidth();

                if ( lwidth < rwidth )
                {
                    lhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, rTyF, lhs );
                }
                else if ( rwidth < lwidth )
                {
                    rhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, lTyF, rhs );
                }

                auto cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
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
        LoadOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::LoadOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto loadOp = cast<toy::LoadOp>( op );
            auto loc = loadOp.getLoc();

            // %0 = toy.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.load: " << *op << '\n' );

            auto varName = loadOp.getVarNameAttr().getRootReference().getValue().str();
            auto allocaOp = lState.lookupLocalSymbolReference( loadOp, varName );

            LLVM_DEBUG( llvm::dbgs() << "varName: " << varName << '\n' );

            Type elemType = allocaOp.getElemType();

            if ( loadOp.getResult().getType() == lState.tyPtr )
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

    class ScopeOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ScopeOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::ScopeOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto scopeOp = cast<toy::ScopeOp>( op );
            auto* funcRegion = scopeOp->getParentRegion();
            if ( !funcRegion || !isa<func::FuncOp>( scopeOp->getParentOp() ) )
            {
                return rewriter.notifyMatchFailure( op, "ScopeOp must be nested in a func::FuncOp" );
            }

            auto* entryBlock = &*funcRegion->begin();
            auto* funcTerminator = entryBlock->getTerminator();

            // Verify that the terminator is a YieldOp
            if ( !isa<toy::YieldOp>( funcTerminator ) )
            {
                return rewriter.notifyMatchFailure( op, "Expected func::FuncOp terminator to be toy::YieldOp" );
            }

            // Erase the YieldOp first to ensure only one terminator will exist
            rewriter.eraseOp( funcTerminator );

            // If ScopeOp has a non-empty body, process its operations
            if ( !scopeOp.getBody().empty() )
            {
                auto& scopeBlock = scopeOp.getBody().front();

                // Set insertion point at the end of the func entry block
                rewriter.setInsertionPointToEnd( entryBlock );

                // Process operations in the scope block
                for ( auto& op : llvm::make_early_inc_range( scopeBlock ) )
                {
                    if ( isa<toy::ReturnOp>( op ) )
                    {
                        // Replace toy::ReturnOp with func::ReturnOp
                        rewriter.create<func::ReturnOp>( op.getLoc(), op.getOperands() );
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
            return success();
        }
    };

    // Now unused (again)
    template <class toyOpType>
    class LowerByDeletion : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        LowerByDeletion( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toyOpType::getOperationName(), benefit, context ), lState{ lState_ }
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

    // Lower toy.print to a call to __toy_print.
    class PrintOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        PrintOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::PrintOp::getOperationName(), benefit, context ), lState{ lState_ }
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
        NegOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( toy::NegOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto unaryOp = cast<toy::NegOp>( op );
            auto loc = unaryOp.getLoc();
            mlir::Value result = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.negate: " << *op << '\n' );

            if ( auto resulti = mlir::dyn_cast<IntegerType>( result.getType() ) )
            {
                result =
                    rewriter.create<LLVM::SubOp>( loc, lState.getIzero( loc, rewriter, resulti.getWidth() ), result );
            }
            else if ( auto resultf = mlir::dyn_cast<FloatType>( result.getType() ) )
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

                result = rewriter.create<LLVM::FSubOp>( loc, lState.getFzero( loc, rewriter, w ), result );
            }
            else
            {
                llvm_unreachable( "Unknown type in negation operation lowering." );
            }

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
        BinaryOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( ToyBinaryOpType::getOperationName(), benefit, context ), lState{ lState_ }
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

                if ( auto lTyI = mlir::dyn_cast<IntegerType>( lhs.getType() ) )
                {
                    auto width = lTyI.getWidth();

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
                        lhs = rewriter.create<LLVM::FPToSIOp>( loc, resultType, lhs );
                    }
                    else
                    {
                        llvm_unreachable( "float types unsupported for integer binary operation" );
                    }
                }

                if ( auto rTyI = mlir::dyn_cast<IntegerType>( rhs.getType() ) )
                {
                    auto width = rTyI.getWidth();

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
                if ( auto lTyI = mlir::dyn_cast<IntegerType>( lhs.getType() ) )
                {
                    auto width = lTyI.getWidth();

                    if ( width == 1 )
                    {
                        lhs = rewriter.create<LLVM::UIToFPOp>( loc, resultType, lhs );
                    }
                    else
                    {
                        lhs = rewriter.create<LLVM::SIToFPOp>( loc, resultType, lhs );
                    }
                }
                if ( auto rTyI = mlir::dyn_cast<IntegerType>( rhs.getType() ) )
                {
                    auto width = rTyI.getWidth();

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
        ConstantOpLowering( loweringContext& lState_, MLIRContext* context, PatternBenefit benefit )
            : ConversionPattern( arith::ConstantOp::getOperationName(), benefit, context ), lState{ lState_ }
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
                auto value = rewriter.create<LLVM::ConstantOp>( loc, lState.tyF64, fAttr );
                rewriter.replaceOp( op, value );
                return success();
            }
            else if ( auto intAttr = dyn_cast<IntegerAttr>( valueAttr ) )
            {
                auto value = rewriter.create<LLVM::ConstantOp>( loc, lState.tyI64, intAttr );
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
            ModuleOp module = getOperation();
            LLVM_DEBUG( {
                llvm::dbgs() << "Starting ToyToLLVMLoweringPass on:\n";
                module->dump();
            } );

            loweringContext lState( module, *pDriverState );
            lState.createDICompileUnit();

            for ( auto funcOp : module.getBodyRegion().getOps<mlir::func::FuncOp>() )
            {
                LLVM_DEBUG( {
                    llvm::dbgs() << "Generating !DISubroutineType() for mlir::func::FuncOp: " << funcOp.getSymName()
                                 << "\n";
                } );
                lState.createFuncDebug( funcOp );
            }

            // First phase: Lower toy operations except ScopeOp and YieldOp
            {
                ConversionTarget target( getContext() );
                target.addLegalDialect<LLVM::LLVMDialect, toy::ToyDialect>();
                target.addIllegalOp<arith::ConstantOp, toy::AddOp, toy::AndOp, toy::AssignOp, toy::DeclareOp,
                                    toy::DivOp, toy::EqualOp, toy::LessEqualOp, toy::LessOp, toy::LoadOp, toy::MulOp,
                                    toy::NegOp, toy::NotEqualOp, toy::OrOp, toy::PrintOp, toy::StringLiteralOp,
                                    toy::SubOp, toy::XorOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp,
                                  toy::ScopeOp, toy::YieldOp, toy::ReturnOp>();

                RewritePatternSet patterns( &getContext() );
                patterns.add<AddOpLowering, AndOpLowering, AssignOpLowering, ConstantOpLowering, DeclareOpLowering,
                             DivOpLowering, EqualOpLowering, LessEqualOpLowering, LessOpLowering, LoadOpLowering,
                             MulOpLowering, NegOpLowering, NotEqualOpLowering, OrOpLowering, PrintOpLowering,
                             StringLiteralOpLowering, SubOpLowering, XorOpLowering>( lState, &getContext(), 1 );
                arith::populateArithToLLVMConversionPatterns( lState.typeConverter, patterns );

                if ( failed( applyFullConversion( module, target, std::move( patterns ) ) ) )
                {
                    LLVM_DEBUG( llvm::dbgs() << "Toy Lowering: First phase failed\n" );
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG( {
                    llvm::dbgs() << "After first phase (toy ops lowered):\n";
                    module->dump();
                } );
            }

            // Second phase: Inline ScopeOp and erase YieldOp
            {
                ConversionTarget target( getContext() );
                target.addLegalDialect<LLVM::LLVMDialect>();
                target.addIllegalOp<toy::ScopeOp, toy::YieldOp, toy::ReturnOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp>();

                RewritePatternSet patterns( &getContext() );
                patterns.add<ScopeOpLowering>( lState, &getContext(), 1 );

                if ( failed( applyFullConversion( module, target, std::move( patterns ) ) ) )
                {
                    LLVM_DEBUG( llvm::dbgs() << "Toy Lowering: Second phase failed\n" );
                    signalPassFailure();
                    return;
                }
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After successful ToyToLLVMLoweringPass:\n";
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
