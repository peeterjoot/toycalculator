/**
 * @file    prototypes/mlirtest.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   This is a standalone MLIR -> LLVM-IR generation program, with working DWARF instrumentation, but done wrong.
 *
 * @section Description
 *
 * I wanted a working standalone MLIR -> LLVM-IR file that was standalone, for which line and variable debugging worked
 * end to end. Insertion of the DI info after the fact is wholely and disgustingly wrong.  This shouldn't be required.
 * I want to use this program as the starting point to figure out how to do this the right way -- i.e.: all the location
 * info that is saved in the MLIR layer shouldn't be thrown away (foracbly reconstructed after the fact), so what is the
 * right way to make sure that lowering converts that into proper DIbuilder statements without this hacking?
 *
 */

#include <assert.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#define FAKE_PROGRAM_SOURCE_NAME "mlirtest.proto"

class myDIBuilder
{
   public:
    llvm::DIBuilder diBuilder;

    // The memory for all these pointers is tied to the DIBuilder builder above, and gets freed at the finalize call
    // later:
    llvm::DIFile *file;
    llvm::DICompileUnit *cu;
    llvm::DISubroutineType *subprogramType;
    llvm::DISubprogram *subprogram;
    llvm::DIType *diTypeI64;
    llvm::Function *llvmFunc;

    llvm::BasicBlock &entryBlock;
    llvm::IRBuilder<> llvmBuilder;

    myDIBuilder( std::unique_ptr<llvm::Module> &llvmModule )
        : diBuilder{ *llvmModule },
          file{ diBuilder.createFile( FAKE_PROGRAM_SOURCE_NAME, "" ) },
          cu{ diBuilder.createCompileUnit( llvm::dwarf::DW_LANG_C, file, "MLIR Compiler", false, "", 0 ) },
          subprogramType{ diBuilder.createSubroutineType( diBuilder.getOrCreateTypeArray( {} ) ) },
          subprogram{ diBuilder.createFunction( file, "main", "main", file, 1, subprogramType, 1,
                                                llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition ) },
          diTypeI64{ diBuilder.createBasicType( "int", 64, llvm::dwarf::DW_ATE_signed ) },
          llvmFunc{ llvmModule->getFunction( "main" ) },
          entryBlock{ llvmFunc->getEntryBlock() },
          llvmBuilder{ &entryBlock }
    {
    }

    void instrumentVariable( const char *varname, unsigned line, unsigned column, llvm::Value *allocaPtr )
    {
        auto *var =
            diBuilder.createAutoVariable( subprogram, varname, file, line, diTypeI64, true /* force emission */ );

        auto *dbgLoc = llvm::DILocation::get( allocaPtr->getContext(), line, 3, subprogram );

        llvmBuilder.SetInsertPoint( &entryBlock, entryBlock.begin() );

        diBuilder.insertDeclare( allocaPtr, var, diBuilder.createExpression(), dbgLoc, llvmBuilder.GetInsertBlock() );
    }
};

// 2:   x = 6;
// 3:   PRINT x;
void buildAssignmentAndPrint( mlir::OpBuilder &builder, mlir::MLIRContext *context, unsigned aline, int value )
{
    auto varLoc = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), aline, 3 );
    auto printLoc = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), aline + 1, 3 );

    mlir::Type int64Type = mlir::IntegerType::get( context, 64 );
    mlir::Type pointerType = mlir::LLVM::LLVMPointerType::get( context );

    mlir::Value i64One = builder.create<mlir::LLVM::ConstantOp>( varLoc, int64Type, builder.getI64IntegerAttr( 1 ) );
    mlir::Value constantValue = builder.create<mlir::arith::ConstantOp>( varLoc, builder.getI64IntegerAttr( value ) );
    mlir::Value ptr = builder.create<mlir::LLVM::AllocaOp>( varLoc, pointerType, int64Type, i64One, 4 );
    builder.create<mlir::LLVM::StoreOp>( varLoc, constantValue, ptr );

    mlir::Value val = builder.create<mlir::LLVM::LoadOp>( printLoc, int64Type, ptr );
    builder.create<mlir::func::CallOp>( printLoc, "__silly_print_i64", mlir::TypeRange{}, mlir::ValueRange{ val } );
}

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Standalone MLIR to LLVM-IR code (builder+lowering)\n" );

    // Initialize MLIR context
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();

    // Register dialect translations
    mlir::registerBuiltinDialectTranslation( context );
    mlir::registerLLVMDialectTranslation( context );

    // Create module
    mlir::OpBuilder builder( &context );
    auto loc = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), 1, 1 );
    auto module = mlir::ModuleOp::create( loc );

    auto printType = builder.getFunctionType( builder.getI64Type(), {} );
    mlir::func::FuncOp print = builder.create<mlir::func::FuncOp>( loc, "__silly_print_i64", printType,
                                                                   llvm::ArrayRef<mlir::NamedAttribute>{} );    //
    print.setPrivate();    // External linkage
    module.push_back( print );

    auto funcType = builder.getFunctionType( {}, builder.getI32Type() );
    auto func = builder.create<mlir::func::FuncOp>( loc, "main", funcType );
    auto &block = *func.addEntryBlock();
    builder.setInsertionPointToStart( &block );

    // 2:   x = 6;
    // 3:   PRINT x;
    buildAssignmentAndPrint( builder, &context, 2, 6 );

    // 4:   y = 7;
    // 5:   PRINT y;
    buildAssignmentAndPrint( builder, &context, 4, 7 );

    // 6:   z = 42;
    // 7:   PRINT z;
    buildAssignmentAndPrint( builder, &context, 6, 42 );

    // 8:   RETURN;
    auto retLocR = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), 8, 3 );
    auto zero32 =
        builder.create<mlir::arith::ConstantOp>( retLocR, builder.getI32Type(), builder.getI32IntegerAttr( 0 ) );
    builder.create<mlir::func::ReturnOp>( retLocR, mlir::ValueRange{ zero32 } );

    // Add function to module
    module.push_back( func );

    // Dump module before passes
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm().enableDebugInfo( true );
    module.print( llvm::outs(), flags );

    // Run passes to lower func.func and arith.constant to LLVMDialect
    mlir::PassManager pm( &context );
    pm.addPass( mlir::createConvertFuncToLLVMPass() );
    pm.addPass( mlir::createArithToLLVMConversionPass() );
    pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
    if ( failed( pm.run( module ) ) )
    {
        llvm::errs() << "Failed to run conversion passes\n";
        return 1;
    }

    // Dump module after passes
    module.print( llvm::outs(), flags );

    // Translate to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR( module, llvmContext, "example" );
    if ( !llvmModule )
    {
        llvm::errs() << "Failed to translate to LLVM IR\n";
        return 1;
    }

    // Add DWARF debug info flags
    // llvmModule->addModuleFlag( llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION );
    llvmModule->addModuleFlag( llvm::Module::Warning, "Dwarf Version", 5 );    // Match Clang's DWARF 5

    // Create debug metadata with DIBuilder
    myDIBuilder dbi( llvmModule );
    auto mainFunc = llvmModule->getFunction( "main" );
    mainFunc->setSubprogram( dbi.subprogram );

    // Add debug locations to instructions
    int assignIndex = 0;
    int printIndex = 0;
    for ( auto &block : *mainFunc )
    {
        for ( auto &inst : block )
        {
            if ( ( inst.getOpcode() == llvm::Instruction::Call ) || ( inst.getOpcode() == llvm::Instruction::Load ) )
            {
                int line = 0;
                switch ( printIndex )
                {
                    case 0:
                        line = 3;    // print(x)
                        break;
                    case 1:
                        line = 5;    // print(y)
                        break;
                    case 2:
                        line = 7;    // print(z)
                        break;
                }
                assert( line );
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, line, 3, dbi.subprogram ) );

                if ( inst.getOpcode() == llvm::Instruction::Call )
                {
                    printIndex++;
                }
            }
            else if ( ( inst.getOpcode() == llvm::Instruction::Alloca ) ||
                      ( inst.getOpcode() == llvm::Instruction::Store ) )
            {
                int line = 0;
                const char *v{};
                switch ( assignIndex )
                {
                    case 0:
                        v = "x";
                        line = 2;    // 2:   x = 6;
                        break;
                    case 1:
                        v = "y";
                        line = 4;    // 4:   y = 7;
                        break;
                    case 2:
                        v = "z";
                        line = 6;    // 6:   z = 42;
                        break;
                }
                assert( v );
                assert( line );
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, line, 3, dbi.subprogram ) );

                if ( inst.getOpcode() == llvm::Instruction::Store )
                {
                    assignIndex++;
                }
                else
                {
                    dbi.instrumentVariable( v, line, 3, &inst );
                }
            }
            else if ( inst.getOpcode() == llvm::Instruction::Ret )
            {
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, 8, 3, dbi.subprogram ) );    // return
            }
            else
            {
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, 1, 1, dbi.subprogram ) );    // default
            }
        }
    }

    // Finalize debug info
    dbi.diBuilder.finalize();

    // Print LLVM IR
    llvmModule->print( llvm::outs(), nullptr, true );

    // Save LLVM IR to file
    std::error_code EC;
    llvm::raw_fd_ostream os( "output.ll", EC );
    llvmModule->print( os, nullptr );
    os.close();

    return 0;
}

// vim: et ts=4 sw=4
