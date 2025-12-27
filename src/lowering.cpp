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
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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
#include "lowering.hpp"

#define DEBUG_TYPE "toy-lowering"

// for llvm.ident and DICompileUnitAttr
#define COMPILER_NAME "toycalculator"

/// For llvm.ident
#define COMPILER_VERSION " V6"

namespace toy
{
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

    class loweringContext
    {
       private:
        mlir::LLVM::DIFileAttr pr_fileAttr;
        std::unordered_map<std::string, mlir::LLVM::DISubprogramAttr> pr_subprogramAttr;
        std::unordered_map<std::string, mlir::LLVM::GlobalOp> pr_stringLiterals;
        mlir::func::FuncOp pr_printFuncF64;
        mlir::func::FuncOp pr_printFuncI64;
        mlir::func::FuncOp pr_printFuncString;
        mlir::func::FuncOp pr_getFuncI1;
        mlir::func::FuncOp pr_getFuncI8;
        mlir::func::FuncOp pr_getFuncI16;
        mlir::func::FuncOp pr_getFuncI32;
        mlir::func::FuncOp pr_getFuncI64;
        mlir::func::FuncOp pr_getFuncF32;
        mlir::func::FuncOp pr_getFuncF64;

        const toy::driverState& pr_driverState;
        mlir::ModuleOp& pr_module;
        mlir::OpBuilder pr_builder;
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
        mlir::LLVMTypeConverter typeConverter;

        loweringContext( mlir::ModuleOp& module, const toy::driverState& driverState )
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

            mlir::MLIRContext* context = pr_builder.getContext();
            tyPtr = mlir::LLVM::LLVMPointerType::get( context );

            tyVoid = mlir::LLVM::LLVMVoidType::get( context );
        }

        unsigned preferredTypeAlignment( mlir::Operation* op, mlir::Type elemType )
        {
            mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
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

        mlir::LLVM::ConstantOp getI8zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI8, rewriter.getI8IntegerAttr( 0 ) );
        }

        mlir::LLVM::ConstantOp getI16zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI16, rewriter.getI16IntegerAttr( 0 ) );
        }

        mlir::LLVM::ConstantOp getI32zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI32, rewriter.getI32IntegerAttr( 0 ) );
        }

        mlir::LLVM::ConstantOp getI64zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( 0 ) );
        }

        /// Returns a cached zero constant for the given integer width (i8, i16, i32, i64).
        /// Throws an exception for unsupported widths.
        mlir::LLVM::ConstantOp getIzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, unsigned width )
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

        mlir::LLVM::ConstantOp getI64one( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( 1 ) );
        }

        mlir::LLVM::ConstantOp getF32zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyF32, rewriter.getF32FloatAttr( 0 ) );
        }

        /// Returns a cached zero constant for the given float width (f32, f64).
        /// Throws an exception for unsupported widths.
        mlir::LLVM::ConstantOp getF64zero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter )
        {
            return rewriter.create<mlir::LLVM::ConstantOp>( loc, tyF64, rewriter.getF64FloatAttr( 0 ) );
        }

        mlir::LLVM::ConstantOp getFzero( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, unsigned width )
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
            mlir::OpBuilder& builder;
            mlir::OpBuilder::InsertPoint oldIP;

           public:
            useModuleInsertionPoint( mlir::ModuleOp& module, mlir::OpBuilder& builder_ )
                : builder{ builder_ }, oldIP{ builder.saveInsertionPoint() }
            {
                builder.setInsertionPointToStart( module.getBody() );
            }

            ~useModuleInsertionPoint()
            {
                builder.restoreInsertionPoint( oldIP );
            }
        };

        template <class Ty>
        void createToyPrintPrototype( mlir::func::FuncOp& printOp, Ty type, const char* name )
        {
            if ( !printOp )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                mlir::FunctionType funcType = mlir::FunctionType::get( pr_builder.getContext(), { type }, {} );

                printOp = pr_builder.create<mlir::func::FuncOp>( pr_module.getLoc(), name, funcType );
                printOp.setVisibility( mlir::SymbolTable::Visibility::Private );
            }
        }

        void createToyPrintF64Prototype()
        {
            createToyPrintPrototype( pr_printFuncF64, tyF64, "__toy_print_f64" );
        }

        void createToyPrintI64Prototype()
        {
            createToyPrintPrototype( pr_printFuncI64, tyI64, "__toy_print_i64" );
        }

        void createToyPrintStringPrototype()
        {
            if ( !pr_printFuncString )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                mlir::FunctionType funcType = mlir::FunctionType::get( pr_builder.getContext(), { tyI64, tyPtr }, {} );
                pr_printFuncString =
                    pr_builder.create<mlir::func::FuncOp>( pr_module.getLoc(), "__toy_print_string", funcType );
                pr_printFuncString.setVisibility( mlir::SymbolTable::Visibility::Private );
            }
        }

        template <class RetTy>
        void createToyGetPrototype( mlir::func::FuncOp& getOp, RetTy retType, const char* name )
        {
            if ( !getOp )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                mlir::FunctionType funcType = mlir::FunctionType::get( pr_builder.getContext(), {},    // no arguments
                                                                       { retType }    // single return type
                );

                getOp = pr_builder.create<mlir::func::FuncOp>( pr_module.getLoc(), name, funcType );
                getOp.setVisibility( mlir::SymbolTable::Visibility::Private );
            }
        }

        void createToyGetI1Prototype()
        {
            // returns int8_t, but checks input to verify 0/1 value.
            createToyGetPrototype( pr_getFuncI1, tyI8, "__toy_get_i1" );
        }

        void createToyGetI8Prototype()
        {
            createToyGetPrototype( pr_getFuncI8, tyI8, "__toy_get_i8" );
        }

        void createToyGetI16Prototype()
        {
            createToyGetPrototype( pr_getFuncI16, tyI16, "__toy_get_i16" );
        }

        void createToyGetI32Prototype()
        {
            createToyGetPrototype( pr_getFuncI32, tyI32, "__toy_get_i32" );
        }

        void createToyGetI64Prototype()
        {
            createToyGetPrototype( pr_getFuncI64, tyI64, "__toy_get_i64" );
        }

        void createToyGetF32Prototype()
        {
            createToyGetPrototype( pr_getFuncF32, tyF32, "__toy_get_f32" );
        }

        void createToyGetF64Prototype()
        {
            createToyGetPrototype( pr_getFuncF64, tyF64, "__toy_get_f64" );
        }

        void createDICompileUnit()
        {
            if ( pr_driverState.wantDebug )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                mlir::MLIRContext* context = pr_builder.getContext();


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
                mlir::DistinctAttr distinctAttr = mlir::DistinctAttr::create( pr_builder.getUnitAttr() );
                pr_compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
                    context, distinctAttr, llvm::dwarf::DW_LANG_C, pr_fileAttr,
                    pr_builder.getStringAttr( COMPILER_NAME ), false, mlir::LLVM::DIEmissionKind::Full,
                    mlir::LLVM::DINameTableKind::Default );
            }

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
            mlir::FunctionType funcType = funcOp.getFunctionType();

            mlir::SmallVector<mlir::LLVM::DITypeAttr> paramTypes;

            mlir::LLVM::DITypeAttr returnType =
                getDIType( funcType.getResults().empty() ? mlir::Type() : funcType.getResults()[0] );
            paramTypes.push_back( returnType );

            for ( mlir::Type argType : funcType.getInputs() )
            {
                paramTypes.push_back( getDIType( argType ) );
            }

            mlir::MLIRContext* context = pr_builder.getContext();

            return mlir::LLVM::DISubroutineTypeAttr::get( context, llvm::DINode::FlagZero, paramTypes );
        }

        void createFuncDebug( mlir::func::FuncOp funcOp )
        {
            if ( pr_driverState.wantDebug )
            {
                useModuleInsertionPoint ip( pr_module, pr_builder );

                mlir::MLIRContext* context = pr_builder.getContext();
                std::string funcName = funcOp.getSymName().str();

                mlir::LLVM::DISubroutineTypeAttr subprogramType = createDISubroutineType( funcOp );

                mlir::LLVM::DISubprogramAttr sub = mlir::LLVM::DISubprogramAttr::get(
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

            std::string funcName = funcOp->getName().getStringRef().str();

            std::string funcNameAndVarName = funcName + "::" + varName;

            mlir::Operation* alloca = pr_symbolToAlloca[funcNameAndVarName];
            return mlir::dyn_cast<mlir::LLVM::AllocaOp>( alloca );
        }

        void createLocalSymbolReference( mlir::LLVM::AllocaOp allocaOp, const std::string& varName )
        {
            mlir::func::FuncOp funcOp = getEnclosingFuncOp( allocaOp );
            std::string funcName = funcOp->getName().getStringRef().str();

            std::string funcNameAndVarName = funcName + "::" + varName;

            pr_symbolToAlloca[funcNameAndVarName] = allocaOp;
        }

        void constructVariableDI( llvm::StringRef varName, mlir::Type& elemType, mlir::FileLineColLoc loc,
                                  unsigned elemSizeInBits, mlir::LLVM::AllocaOp& allocaOp, int64_t arraySize = 1 )
        {
            if ( pr_driverState.wantDebug )
            {
                mlir::MLIRContext* context = pr_builder.getContext();

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
                mlir::LLVM::DISubprogramAttr sub = pr_subprogramAttr[funcName];
                assert( sub );

                unsigned totalSizeInBits = elemStorageSizeInBits * arraySize;
                if ( arraySize > 1 )
                {
                    // Create base type for array elements
                    mlir::LLVM::DIBasicTypeAttr baseType = mlir::LLVM::DIBasicTypeAttr::get(
                        context, llvm::dwarf::DW_TAG_base_type, pr_builder.getStringAttr( typeName ),
                        elemStorageSizeInBits, dwType );

                    // Create subrange for array (count = arraySize, lowerBound = 0)
                    mlir::IntegerAttr countAttr = mlir::IntegerAttr::get( tyI64, arraySize );
                    mlir::IntegerAttr lowerBoundAttr = mlir::IntegerAttr::get( tyI64, 0 );
                    mlir::LLVM::DISubrangeAttr subrange =
                        mlir::LLVM::DISubrangeAttr::get( context, countAttr, lowerBoundAttr,
                                                         /*upperBound=*/nullptr, /*stride=*/nullptr );

                    // Create array type
                    unsigned alignInBits = elemStorageSizeInBits;    // Alignment matches element size
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

        mlir::LLVM::GlobalOp lookupOrInsertGlobalOp( mlir::ConversionPatternRewriter& rewriter,
                                                     mlir::StringAttr& stringLit, mlir::Location loc, size_t strLen )
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
                mlir::OpBuilder::InsertPoint savedIP = rewriter.saveInsertionPoint();
                rewriter.setInsertionPointToStart( pr_module.getBody() );

                mlir::LLVM::LLVMArrayType arrayType = mlir::LLVM::LLVMArrayType::get( tyI8, strLen );

                mlir::SmallVector<char> stringData( stringLit.begin(), stringLit.end() );
                mlir::DenseElementsAttr denseAttr = mlir::DenseElementsAttr::get(
                    mlir::RankedTensorType::get( { static_cast<int64_t>( strLen ) }, tyI8 ),
                    mlir::ArrayRef<char>( stringData ) );

                std::string globalName = "str_" + std::to_string( pr_stringLiterals.size() );
                globalOp = rewriter.create<mlir::LLVM::GlobalOp>( loc, arrayType, true, mlir::LLVM::Linkage::Private,
                                                                  globalName, denseAttr );
                globalOp->setAttr( "unnamed_addr", rewriter.getUnitAttr() );

                pr_stringLiterals[stringLit.str()] = globalOp;
                LLVM_DEBUG( llvm::dbgs() << "Created global: " << globalName << '\n' );

                rewriter.restoreInsertionPoint( savedIP );
            }

            return globalOp;
        }

        toy::CallOp createPrintCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::Value input )
        {
            mlir::Type inputType = input.getType();
            toy::CallOp result;

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

                createToyPrintI64Prototype();
                result = rewriter.create<toy::CallOp>( loc, mlir::TypeRange{}, "__toy_print_i64",
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

                createToyPrintF64Prototype();
                result = rewriter.create<toy::CallOp>( loc, mlir::TypeRange{}, "__toy_print_f64",
                                                       mlir::ValueRange{ input } );
            }
            else if ( inputType == tyPtr )
            {
                // Find AllocaOp for size and element type
                int64_t numElems = 0;
                if ( toy::LoadOp loadOp = input.getDefiningOp<toy::LoadOp>() )
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

                    if ( mlir::LLVM::ConstantOp constOp =
                             allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
                    {
                        mlir::IntegerAttr intAttr = mlir::dyn_cast<mlir::IntegerAttr>( constOp.getValue() );
                        numElems = intAttr.getInt();
                    }
                }
                else if ( toy::StringLiteralOp stringLitOp = input.getDefiningOp<toy::StringLiteralOp>() )
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
                        // mlir::ModuleOp module = input.getParentOfType<mlir::ModuleOp>();
                        // module.dump();
                    } );

                    assert( 0 );    // should not get here.
                }
                assert( numElems );

                mlir::LLVM::ConstantOp sizeConst =
                    rewriter.create<mlir::LLVM::ConstantOp>( loc, tyI64, rewriter.getI64IntegerAttr( numElems ) );

                createToyPrintStringPrototype();
                const char* name = "__toy_print_string";
                result =
                    rewriter.create<toy::CallOp>( loc, mlir::TypeRange{}, name, mlir::ValueRange{ sizeConst, input } );
            }
            else
            {
                assert( 0 );    // Error: unsupported type
            }

#if 0
            LLVM_DEBUG( {
                llvm::errs() << "######################### module dump after print lowering\n";
                mlir::ModuleOp module = result->getParentOfType<mlir::ModuleOp>();
                module.dump();
            } );
#endif

            return result;
        }

        mlir::Value createGetCall( mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::Type inputType )
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
                        name = "__toy_get_i1";
                        createToyGetI1Prototype();
                        inputType = tyI8;
                        isBool = true;
                        break;
                    }
                    case 8:
                    {
                        name = "__toy_get_i8";
                        createToyGetI8Prototype();
                        break;
                    }
                    case 16:
                    {
                        name = "__toy_get_i16";
                        createToyGetI16Prototype();
                        break;
                    }
                    case 32:
                    {
                        name = "__toy_get_i32";
                        createToyGetI32Prototype();
                        break;
                    }
                    case 64:
                    {
                        name = "__toy_get_i64";
                        createToyGetI64Prototype();
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
                    name = "__toy_get_f32";
                    createToyGetF32Prototype();
                }
                else if ( inputType == tyF64 )
                {
                    name = "__toy_get_f64";
                    createToyGetF64Prototype();
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

            toy::CallOp callOp =
                rewriter.create<toy::CallOp>( loc, mlir::TypeRange{ inputType }, name, mlir::ValueRange{} );
            mlir::Value result = *callOp.getResult().begin();

            if ( isBool )
            {
                result = rewriter.create<mlir::LLVM::TruncOp>( loc, tyI1, result );
            }

            return result;
        }

        /// Emit debug info for parameter
        void constructParameterDI( mlir::FileLineColLoc loc, mlir::ConversionPatternRewriter& rewriter,
                                   const std::string& varName, mlir::LLVM::AllocaOp value, mlir::Type elemType,
                                   int paramIndex, const std::string& funcName )
        {
            if ( pr_driverState.wantDebug )
            {
                // Create debug type for basic types (e.g., i32, f32)
                mlir::MLIRContext* context = rewriter.getContext();
                mlir::LLVM::DITypeAttr diType = getDIType( elemType );

                // Get DISubprogram from pr_subprogramAttr
                mlir::LLVM::DISubprogramAttr sub = pr_subprogramAttr[funcName];
                assert( sub );

                unsigned bitWidth = elemType.getIntOrFloatBitWidth();

                // Create debug variable
                mlir::LLVM::DILocalVariableAttr diVar = mlir::LLVM::DILocalVariableAttr::get(
                    context, sub, rewriter.getStringAttr( varName ), pr_fileAttr, loc.getLine(), paramIndex + 1,
                    bitWidth, diType, mlir::LLVM::DIFlags::Zero );

                // Emit llvm.dbg.declare
                rewriter.create<mlir::LLVM::DbgDeclareOp>( loc, value, diVar );
            }
        }

        mlir::Value castToElemType( mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, mlir::Value value,
                                    mlir::Type valType, mlir::Type elemType )
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
    };

    class DeclareOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        DeclareOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::DeclareOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::DeclareOp declareOp = cast<toy::DeclareOp>( op );
            mlir::Location loc = declareOp.getLoc();
            bool param = declareOp.isParameter();

            //   toy.declare "x" : i32
            LLVM_DEBUG( llvm::dbgs() << std::format( "Lowering toy.declare: param: {}", param ) << declareOp << '\n' );

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
        loweringContext& lState;

       public:
        StringLiteralOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::StringLiteralOp::getOperationName(), benefit, context ), lState( lState_ )
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::StringLiteralOp stringLiteralOp = cast<toy::StringLiteralOp>( op );
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
        loweringContext& lState;

       public:
        AssignOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::AssignOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::AssignOp assignOp = cast<toy::AssignOp>( op );
            mlir::Location loc = assignOp.getLoc();

            // (ins StrAttr:$name, AnyType:$value);
            // toy.assign "x", %0 : i32
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
            else if ( toy::StringLiteralOp stringLitOp = value.getDefiningOp<toy::StringLiteralOp>() )
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
                auto optIndex = assignOp.getIndex();    // std::optional<Value>

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
    template <class ToyOp, class IOpType, class FOpType, mlir::LLVM::ICmpPredicate ICmpPredS,
              mlir::LLVM::ICmpPredicate ICmpPredU, mlir::LLVM::FCmpPredicate FCmpPred>
    class ComparisonOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ComparisonOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( ToyOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            ToyOp compareOp = cast<ToyOp>( op );
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
        ComparisonOpLowering<toy::LessOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::slt,
                             mlir::LLVM::ICmpPredicate::ult, mlir::LLVM::FCmpPredicate::olt>;

    using LessEqualOpLowering =
        ComparisonOpLowering<toy::LessEqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::sle,
                             mlir::LLVM::ICmpPredicate::ule, mlir::LLVM::FCmpPredicate::ole>;

    using EqualOpLowering =
        ComparisonOpLowering<toy::EqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::eq,
                             mlir::LLVM::ICmpPredicate::eq, mlir::LLVM::FCmpPredicate::oeq>;

    using NotEqualOpLowering =
        ComparisonOpLowering<toy::NotEqualOp, mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp, mlir::LLVM::ICmpPredicate::ne,
                             mlir::LLVM::ICmpPredicate::ne, mlir::LLVM::FCmpPredicate::one>;

    class LoadOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        LoadOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::LoadOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::LoadOp loadOp = cast<toy::LoadOp>( op );
            mlir::Location loc = loadOp.getLoc();

            // %0 = toy.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.load: " << *op << '\n' );

            std::string varName = loadOp.getVarNameAttr().getRootReference().getValue().str();
            mlir::LLVM::AllocaOp allocaOp = lState.lookupLocalSymbolReference( loadOp, varName );
            auto optIndex = loadOp.getIndex();

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
        loweringContext& lState;

       public:
        CallOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::CallOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::CallOp callOp = cast<toy::CallOp>( op );
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
        loweringContext& lState;

       public:
        ScopeOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::ScopeOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::ScopeOp scopeOp = cast<toy::ScopeOp>( op );
            mlir::Region* funcRegion = scopeOp->getParentRegion();
            if ( !funcRegion || !isa<mlir::func::FuncOp>( scopeOp->getParentOp() ) )
            {
                return rewriter.notifyMatchFailure( op, "ScopeOp must be nested in a func::FuncOp" );
            }

            mlir::Block* entryBlock = &*funcRegion->begin();
            mlir::Operation* funcTerminator = entryBlock->getTerminator();

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
                mlir::Block& scopeBlock = scopeOp.getBody().front();

                // Set insertion point at the end of the func entry block
                rewriter.setInsertionPointToEnd( entryBlock );

                // Process operations in the scope block
                for ( mlir::Operation& op : llvm::make_early_inc_range( scopeBlock ) )
                {
                    if ( isa<toy::ReturnOp>( op ) )
                    {
                        // Replace toy::ReturnOp with func::ReturnOp
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
    template <class toyOpType>
    class LowerByDeletion : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        LowerByDeletion( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toyOpType::getOperationName(), benefit, context ), lState{ lState_ }
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

    // Lower toy.print to a call to __toy_print.
    class PrintOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        PrintOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::PrintOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
#if 0
            mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
#endif

            toy::PrintOp printOp = cast<toy::PrintOp>( op );
            mlir::Location loc = printOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.print: " << *op << '\n' );

            mlir::Value input = printOp.getInput();
            LLVM_DEBUG( llvm::dbgs() << "input: " << input << '\n' );

            toy::CallOp result = lState.createPrintCall( rewriter, loc, input );

            rewriter.replaceOp( op, result );

#if 0
            LLVM_DEBUG( {
                llvm::errs() << "######################### module dump after PrintOpLowering::matchAndRewrite replaceOp\n";
                module.dump();
            } );
#endif

            return mlir::success();
        }
    };

    class GetOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        GetOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::GetOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::GetOp getOp = cast<toy::GetOp>( op );
            mlir::Location loc = getOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.get: " << *op << '\n' );

            mlir::Type inputType = getOp.getValue().getType();

            mlir::Value result = lState.createGetCall( rewriter, loc, inputType );

            rewriter.replaceOp( op, result );

            return mlir::success();
        }
    };

    // Lower toy.negate to LLVM arithmetic.
    class NegOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        NegOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( toy::NegOp::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            toy::NegOp negOp = cast<toy::NegOp>( op );
            mlir::Location loc = negOp.getLoc();
            mlir::Value result = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.negate: " << *op << '\n' );

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

    // Lower toy.binary to LLVM arithmetic.
    template <class ToyBinaryOpType, class llvmIOpType, class llvmFOpType, bool allowFloat>
    class BinaryOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        BinaryOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( ToyBinaryOpType::getOperationName(), benefit, context ), lState{ lState_ }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            ToyBinaryOpType binaryOp = cast<ToyBinaryOpType>( op );
            mlir::Location loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.binary: " << *op << '\n' );

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

    using AddOpLowering = BinaryOpLowering<toy::AddOp, mlir::LLVM::AddOp, mlir::LLVM::FAddOp, true>;
    using SubOpLowering = BinaryOpLowering<toy::SubOp, mlir::LLVM::SubOp, mlir::LLVM::FSubOp, true>;
    using MulOpLowering = BinaryOpLowering<toy::MulOp, mlir::LLVM::MulOp, mlir::LLVM::FMulOp, true>;
    using DivOpLowering = BinaryOpLowering<toy::DivOp, mlir::LLVM::SDivOp, mlir::LLVM::FDivOp, true>;

    // mlir::LLVM::FAddOp is a dummy operation here, knowing that it will not ever be used:
    using XorOpLowering = BinaryOpLowering<toy::XorOp, mlir::LLVM::XOrOp, mlir::LLVM::FAddOp, false>;
    using AndOpLowering = BinaryOpLowering<toy::AndOp, mlir::LLVM::AndOp, mlir::LLVM::FAddOp, false>;
    using OrOpLowering = BinaryOpLowering<toy::OrOp, mlir::LLVM::OrOp, mlir::LLVM::FAddOp, false>;

    // Lower arith.constant to LLVM constant.
    class ConstantOpLowering : public mlir::ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ConstantOpLowering( loweringContext& lState_, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( mlir::arith::ConstantOp::getOperationName(), benefit, context ),
              lState{ lState_ }
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

    class ToyToLLVMLoweringPass : public mlir::PassWrapper<ToyToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( ToyToLLVMLoweringPass )

        ToyToLLVMLoweringPass( toy::driverState* pst_ ) : pDriverState{ pst_ }
        {
        }

        void getDependentDialects( mlir::DialectRegistry& registry ) const override
        {
            registry.insert<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
        }

        void runOnOperation() override
        {
            mlir::ModuleOp module = getOperation();
            LLVM_DEBUG( {
                llvm::dbgs() << "Starting ToyToLLVMLoweringPass on:\n";
                module->dump();
            } );

            loweringContext lState( module, *pDriverState );
            lState.createDICompileUnit();

            for ( mlir::func::FuncOp funcOp : module.getBodyRegion().getOps<mlir::func::FuncOp>() )
            {
                LLVM_DEBUG( {
                    llvm::dbgs() << "Generating !DISubroutineType() for mlir::func::FuncOp: " << funcOp.getSymName()
                                 << "\n";
                } );
                lState.createFuncDebug( funcOp );
            }

            // First phase: Lower toy operations except ScopeOp and YieldOp
            {
                mlir::ConversionTarget target( getContext() );
                target.addLegalDialect<mlir::LLVM::LLVMDialect, toy::ToyDialect, mlir::scf::SCFDialect>();
                target.addIllegalOp<mlir::arith::ConstantOp, toy::AddOp, toy::AndOp, toy::AssignOp, toy::DeclareOp,
                                    toy::DivOp, toy::EqualOp, toy::LessEqualOp, toy::LessOp, toy::LoadOp, toy::MulOp,
                                    toy::NegOp, toy::NotEqualOp, toy::OrOp, toy::PrintOp, toy::GetOp,
                                    toy::StringLiteralOp, toy::SubOp, toy::XorOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp,
                                  toy::ScopeOp, toy::YieldOp, toy::ReturnOp, toy::CallOp, mlir::func::CallOp,
                                  mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::YieldOp>();

                mlir::RewritePatternSet patterns( &getContext() );
                patterns.add<AddOpLowering, AndOpLowering, AssignOpLowering, ConstantOpLowering, DeclareOpLowering,
                             DivOpLowering, EqualOpLowering, LessEqualOpLowering, LessOpLowering, LoadOpLowering,
                             MulOpLowering, NegOpLowering, NotEqualOpLowering, OrOpLowering, PrintOpLowering,
                             GetOpLowering, StringLiteralOpLowering, SubOpLowering, XorOpLowering>( lState,
                                                                                                    &getContext(), 1 );
                mlir::arith::populateArithToLLVMConversionPatterns( lState.typeConverter, patterns );

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
                mlir::ConversionTarget target( getContext() );
                target.addLegalDialect<mlir::LLVM::LLVMDialect>();
                target.addIllegalOp<toy::ScopeOp, toy::YieldOp, toy::ReturnOp, toy::CallOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp>();
                target.addIllegalDialect<mlir::scf::SCFDialect>();
                target.addIllegalDialect<mlir::cf::ControlFlowDialect>();    // forces lowering

                mlir::RewritePatternSet patterns( &getContext() );
                patterns.add<CallOpLowering, ScopeOpLowering>( lState, &getContext(), 1 );

                // SCF -> CF
                mlir::populateSCFToControlFlowConversionPatterns( patterns );

                mlir::arith::populateArithToLLVMConversionPatterns( lState.typeConverter, patterns );

                // CF -> LLVM
                mlir::cf::populateControlFlowToLLVMConversionPatterns( lState.typeConverter, patterns );

                if ( failed( applyFullConversion( module, target, std::move( patterns ) ) ) )
                {
                    LLVM_DEBUG( llvm::dbgs() << "Toy Lowering: Second phase failed\n" );
                    signalPassFailure();
                    return;
                }
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After successful ToyToLLVMLoweringPass:\n";
                for ( mlir::Operation& op : module->getRegion( 0 ).front() )
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
