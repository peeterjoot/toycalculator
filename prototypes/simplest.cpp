#include <assert.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
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

#include <optional>

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Minimal MLIR to LLVM-IR debug test\n" );

    // Initialize LLVM targets explicitly for X86
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    // Get target triple
    std::string targetTriple = llvm::sys::getDefaultTargetTriple();
    llvm::Triple triple( targetTriple );
    assert( triple.isArch64Bit() && triple.isOSLinux() );

    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget( targetTriple, error );
    assert( target );
    llvm::TargetOptions options;
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>( target->createTargetMachine(
        targetTriple, "generic", "", options, std::optional<llvm::Reloc::Model>( llvm::Reloc::PIC_ ) ) );
    assert( targetMachine );
    std::string dataLayoutStr = targetMachine->createDataLayout().getStringRepresentation();

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    mlir::registerBuiltinDialectTranslation( context );
    mlir::registerLLVMDialectTranslation( context );

    mlir::OpBuilder builder( &context );
    auto loc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), 1, 1 );
    auto module = mlir::ModuleOp::create( loc );
    module->setAttr( "llvm.data_layout", builder.getStringAttr( dataLayoutStr ) );
    module->setAttr( "llvm.ident", builder.getStringAttr( "toycompiler 0.0" ) );
    module->setAttr( "llvm.target_triple", builder.getStringAttr( targetTriple ) );

    auto funcType = builder.getFunctionType( {}, builder.getI32Type() );
    auto func = builder.create<mlir::func::FuncOp>( loc, "main", funcType );
    auto &block = *func.addEntryBlock();
    builder.setInsertionPointToStart( &block );

    // Add variable x
    auto varLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), 2, 3 );
    auto ptrType = mlir::LLVM::LLVMPointerType::get( &context );
    auto one = builder.create<mlir::LLVM::ConstantOp>( varLoc, builder.getI64Type(), builder.getI64IntegerAttr( 1 ) );
    mlir::Type int64Type = mlir::IntegerType::get( &context, 64 );
    auto allocaOp = builder.create<mlir::LLVM::AllocaOp>( varLoc, ptrType, int64Type, one, 4 );

    auto i32Type = builder.getI32Type();
    auto const42 = builder.create<mlir::LLVM::ConstantOp>( varLoc, i32Type, builder.getI32IntegerAttr( 42 ) );
    builder.create<mlir::LLVM::StoreOp>( varLoc, const42, allocaOp );

    auto fileAttr = mlir::LLVM::DIFileAttr::get( &context, "test.c", "." );
    auto distinctAttr = mlir::DistinctAttr::create( builder.getUnitAttr() );
    auto compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
        &context, distinctAttr, llvm::dwarf::DW_LANG_C, fileAttr, builder.getStringAttr( "testcompiler" ), false,
        mlir::LLVM::DIEmissionKind::Full, mlir::LLVM::DINameTableKind::Default );
    auto ta =
        mlir::LLVM::DIBasicTypeAttr::get( &context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                          builder.getStringAttr( "int" ), 32, (unsigned)llvm::dwarf::DW_ATE_signed );
    llvm::SmallVector<mlir::LLVM::DITypeAttr, 1> typeArray;
    typeArray.push_back( ta );
    auto subprogramType = mlir::LLVM::DISubroutineTypeAttr::get( &context, 0, typeArray );
    auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
        &context, mlir::DistinctAttr::create( builder.getUnitAttr() ), compileUnitAttr, fileAttr,
        builder.getStringAttr( "main" ), builder.getStringAttr( "main" ), fileAttr, 1, 1,
        mlir::LLVM::DISubprogramFlags::Definition, subprogramType, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{},
        llvm::ArrayRef<mlir::LLVM::DINodeAttr>{} );
    func->setAttr( "llvm.debug.subprogram", subprogramAttr );
    func->setLoc( builder.getFusedLoc( { loc }, subprogramAttr ) );
    module.push_back( func );

    // Variable DI
    allocaOp->setAttr( "bindc_name", builder.getStringAttr( "x" ) );
    auto diType = mlir::LLVM::DIBasicTypeAttr::get( &context, llvm::dwarf::DW_TAG_base_type,
                                                    builder.getStringAttr( "int" ), 32, llvm::dwarf::DW_ATE_signed );
    auto diVar = mlir::LLVM::DILocalVariableAttr::get( &context, subprogramAttr, builder.getStringAttr( "x" ), fileAttr,
                                                       2, 0, 32, diType, mlir::LLVM::DIFlags::Zero );
    builder.create<mlir::LLVM::DbgDeclareOp>( varLoc, allocaOp, diVar );

    // Return 0
    auto retLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), 3, 3 );
    auto zero = builder.create<mlir::LLVM::ConstantOp>( retLoc, i32Type, builder.getI32IntegerAttr( 0 ) );
    builder.create<mlir::func::ReturnOp>( retLoc, mlir::ValueRange{ zero } );

    // llvm::outs() << "Dump module before passes:\n";
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm().enableDebugInfo( true );
    // module.print( llvm::outs(), flags );

    mlir::PassManager pm( &context );
    pm.addPass( mlir::createConvertFuncToLLVMPass() );
    if ( failed( pm.run( module ) ) )
    {
        llvm::errs() << "Failed to run conversion passes\n";
        return 1;
    }

    // llvm::errs() << "After lowering all but module (before translateModuleToLLVMIR):\n\n";
    // module.print( llvm::outs(), flags );

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR( module, llvmContext, "test" );
    if ( !llvmModule )
    {
        llvm::errs() << "Failed to translate to LLVM IR\n";
        return 1;
    }

    // llvm::errs() << "After lowering\n\n";
    llvmModule->print( llvm::outs(), nullptr, true );
    return 0;
}
