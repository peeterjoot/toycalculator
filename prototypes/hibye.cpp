/**
 * @file    prototypes/hibye.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Standalone MLIR -> LLVM-IR generation program with DWARF instrumentation.
 *
 * @section Description
 *
 * This is a MWE that illustrates a compiler workflow for a simple C program with if-else and printf:
 * - MLIR builder
 * - Lowering (no special lowering classes required.)
 * - LLVM module replacement for top level.
 *
 * Use this to build an LLVM-IR equivalent program to hibye.c with:
 *
    ../build/hibye  > output.ll
    clang -g -o output output.ll -Wno-override-module
    gdb -q ./output
 *
 * Then perform basic debugging operations, including:
 *
    (gdb) b hibye.c:2
    (gdb) b main
    (gdb) run
    (gdb) n
    (gdb) p argc
 *
 */

#include <assert.h>
#include <llvm/ADT/SmallVector.h>
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
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <optional>

// I have no intention of cross compiling and have assumed target-platform=host-platform=linux 64-bit.
// These defines aren't an attempt to make the code portable, assuming that any other target will ever
// be used, but are simply to help give meaning to random 4,8,32,64 values scattered below.
#define TARGET_CHAR_SIZE_IN_BITS 8
#define TARGET_INT_SIZE_IN_BYTES sizeof( int )
#define TARGET_INT_SIZE_IN_BITS ( TARGET_INT_SIZE_IN_BYTES * 8 )
#define TARGET_LONG_SIZE_IN_BYTES sizeof( long )
#define TARGET_LONG_SIZE_IN_BITS ( TARGET_LONG_SIZE_IN_BYTES * 8 )
#define TARGET_POINTER_SIZE_IN_BYTES sizeof( char * )
#define TARGET_POINTER_SIZE_IN_BITS ( TARGET_POINTER_SIZE_IN_BYTES * 8 )

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Minimal MLIR to LLVM-IR debug test\n" );

    // Initialize LLVM targets for X86
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

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
    context.getOrLoadDialect<mlir::arith::ArithDialect>();       // For constants and comparisons
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();    // For conditional branching

    mlir::registerBuiltinDialectTranslation( context );
    mlir::registerLLVMDialectTranslation( context );

/*
1: #include <stdio.h>
2: int main(int argc, char** argv) {
3:     if (argc) {
4:         printf("hi\n");
5:     } else {
6:         printf("bye\n");
7:     }
8:     return 0;
9: }
*/
#define MODULE_LINE 1
#define main_LINE 2
#define if_LINE 3
#define printf_hi_LINE 4
#define else_LINE 5
#define printf_bye_LINE 6
#define return_LINE 7
#define THE_COLUMN_START 3

    mlir::OpBuilder builder( &context );
    auto moduleLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "hibye.c" ), MODULE_LINE, 1 );
    auto module = mlir::ModuleOp::create( moduleLoc );
    module->setAttr( "llvm.data_layout", builder.getStringAttr( dataLayoutStr ) );
    module->setAttr( "llvm.ident", builder.getStringAttr( "toycompiler 0.0" ) );
    module->setAttr( "llvm.target_triple", builder.getStringAttr( targetTriple ) );

    // Create global string constant for "hi\n"
    builder.setInsertionPointToStart( module.getBody() );
    auto i8Type = builder.getI8Type();
    auto hiArrayType = mlir::LLVM::LLVMArrayType::get( i8Type, 4 );    // "hi\n\00"
    llvm::SmallVector<char> hiStringData = { 'h', 'i', '\n', '\0' };
    auto hiDenseAttr = mlir::DenseElementsAttr::get( mlir::RankedTensorType::get( { 4 }, i8Type ),
                                                     llvm::ArrayRef<char>( hiStringData ) );
    auto hiGlobalOp = builder.create<mlir::LLVM::GlobalOp>( moduleLoc, hiArrayType, /*isConstant=*/true,
                                                            mlir::LLVM::Linkage::Private, "str_hi", hiDenseAttr );
    hiGlobalOp->setAttr( "unnamed_addr", builder.getUnitAttr() );

    // Create global string constant for "bye\n"
    auto byeArrayType = mlir::LLVM::LLVMArrayType::get( i8Type, 5 );    // "bye\n\00"
    llvm::SmallVector<char> byeStringData = { 'b', 'y', 'e', '\n', '\0' };
    auto byeDenseAttr = mlir::DenseElementsAttr::get( mlir::RankedTensorType::get( { 5 }, i8Type ),
                                                      llvm::ArrayRef<char>( byeStringData ) );
    auto byeGlobalOp = builder.create<mlir::LLVM::GlobalOp>( moduleLoc, byeArrayType, /*isConstant=*/true,
                                                             mlir::LLVM::Linkage::Private, "str_bye", byeDenseAttr );
    byeGlobalOp->setAttr( "unnamed_addr", builder.getUnitAttr() );

    // Create main function with arguments
    auto mainLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "hibye.c" ), main_LINE, 1 );
    auto i32Type = builder.getI32Type();
    auto ptrType = mlir::LLVM::LLVMPointerType::get( &context );
    auto funcType = builder.getFunctionType( { i32Type, ptrType }, i32Type );    // int argc, char** argv
    auto func = builder.create<mlir::func::FuncOp>( mainLoc, "main", funcType );
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart( &entryBlock );

    // Debug info for the compilation unit and subprogram
    auto fileAttr = mlir::LLVM::DIFileAttr::get( &context, "hibye.c", "." );
    auto distinctAttr = mlir::DistinctAttr::create( builder.getUnitAttr() );
    auto compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
        &context, distinctAttr, llvm::dwarf::DW_LANG_C, fileAttr, builder.getStringAttr( "testcompiler" ), false,
        mlir::LLVM::DIEmissionKind::Full, mlir::LLVM::DINameTableKind::Default );
    auto returnTypeAttr = mlir::LLVM::DIBasicTypeAttr::get( &context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                            builder.getStringAttr( "int" ), TARGET_INT_SIZE_IN_BITS,
                                                            (unsigned)llvm::dwarf::DW_ATE_signed );
    llvm::SmallVector<mlir::LLVM::DITypeAttr, 2> typeArray;
    typeArray.push_back( returnTypeAttr );    // For argc (int)
    typeArray.push_back( returnTypeAttr );    // Placeholder for argv (char**), simplified
    auto subprogramType = mlir::LLVM::DISubroutineTypeAttr::get( &context, 0, typeArray );
    auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
        &context, mlir::DistinctAttr::create( builder.getUnitAttr() ), compileUnitAttr, fileAttr,
        builder.getStringAttr( "main" ), builder.getStringAttr( "main" ), fileAttr, main_LINE, main_LINE,
        mlir::LLVM::DISubprogramFlags::Definition, subprogramType, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{},
        llvm::ArrayRef<mlir::LLVM::DINodeAttr>{} );
    func->setAttr( "llvm.debug.subprogram", subprogramAttr );

    // Debug info for argc
    auto ifLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "hibye.c" ), if_LINE, THE_COLUMN_START );
    auto argcVal = entryBlock.getArgument( 0 );    // First argument: argc
    auto di_argcType =
        mlir::LLVM::DIBasicTypeAttr::get( &context, llvm::dwarf::DW_TAG_base_type, builder.getStringAttr( "int" ),
                                          TARGET_INT_SIZE_IN_BITS, llvm::dwarf::DW_ATE_signed );
    auto di_argcVar = mlir::LLVM::DILocalVariableAttr::get( &context, subprogramAttr, builder.getStringAttr( "argc" ),
                                                            fileAttr, if_LINE, 0, TARGET_INT_SIZE_IN_BITS, di_argcType,
                                                            mlir::LLVM::DIFlags::Zero );
    builder.create<mlir::LLVM::DbgDeclareOp>( ifLoc, argcVal, di_argcVar );

    // If condition: check if argc > 0
    auto zero = builder.create<mlir::arith::ConstantOp>( ifLoc, i32Type, builder.getI32IntegerAttr( 0 ) );
    auto cond = builder.create<mlir::arith::CmpIOp>( ifLoc, mlir::arith::CmpIPredicate::sgt, argcVal, zero );

    // Create blocks for if-then, else, and exit
    auto thenBlock = func.addBlock();
    auto elseBlock = func.addBlock();
    auto exitBlock = func.addBlock();
    builder.create<mlir::cf::CondBranchOp>( ifLoc, cond, thenBlock, mlir::ValueRange{}, elseBlock, mlir::ValueRange{} );

    // Then block: printf("hi\n")
    builder.setInsertionPointToStart( thenBlock );
    auto hiLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "hibye.c" ), printf_hi_LINE, THE_COLUMN_START );
    auto hiGlobalPtr = builder.create<mlir::LLVM::AddressOfOp>( hiLoc, hiGlobalOp );
    builder.create<mlir::LLVM::CallOp>( hiLoc, i32Type, builder.getStringAttr( "printf" ),
                                        mlir::ValueRange{ hiGlobalPtr } );
    builder.create<mlir::cf::BranchOp>( hiLoc, exitBlock, mlir::ValueRange{} );

    // Else block: printf("bye\n")
    builder.setInsertionPointToStart( elseBlock );
    auto byeLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "hibye.c" ), printf_bye_LINE, THE_COLUMN_START );
    auto byeGlobalPtr = builder.create<mlir::LLVM::AddressOfOp>( byeLoc, byeGlobalOp );
    builder.create<mlir::LLVM::CallOp>( byeLoc, i32Type, builder.getStringAttr( "printf" ),
                                        mlir::ValueRange{ byeGlobalPtr } );
    builder.create<mlir::cf::BranchOp>( byeLoc, exitBlock, mlir::ValueRange{} );

    // Exit block: return 0
    builder.setInsertionPointToStart( exitBlock );
    auto retLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "hibye.c" ), return_LINE, THE_COLUMN_START );
    auto retVal = builder.create<mlir::LLVM::ConstantOp>( retLoc, i32Type, builder.getI32IntegerAttr( 0 ) );
    builder.create<mlir::func::ReturnOp>( retLoc, mlir::ValueRange{ retVal } );

    // Set fused location for debug info translation
    func->setLoc( builder.getFusedLoc( { mainLoc }, subprogramAttr ) );

    // Run passes
    mlir::PassManager pm( &context );
    pm.addPass( mlir::createConvertArithToLLVMPass() );
    pm.addPass( mlir::createConvertFuncToLLVMPass() );
    if ( failed( pm.run( module ) ) )
    {
        llvm::errs() << "Failed to run conversion passes\n";
        return 1;
    }

    // Translate to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR( module, llvmContext, "test" );
    if ( !llvmModule )
    {
        llvm::errs() << "Failed to translate to LLVM IR\n";
        return 1;
    }

    llvmModule->print( llvm::outs(), nullptr, true );
    return 0;
}
