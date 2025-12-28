/**
 * @file    prototypes/simplest.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Standalone MLIR -> LLVM-IR generation program with DWARF instrumentation.
 *
 * @section Description
 *
 * Unlike mlirtest.cpp, this emits DI information in the MLIR builder.  It's a MWE that illustrates a
 * compiler workflow:
 * - MLIR builder
 * - Lowering (no special lowering classes required.)
 * - LLVM module replacement for top level.
 *
 * Use this to build an LLVM-IR equivalent program to test.c with:
 *
    ../build/simplest  > output.ll
    clang -g -o output output.ll -Wno-override-module
    gdb -q ./output

    and then some basic debugging operations, including:

    (gdb) b test.c:2
    (gdb) b main
    (gdb) run
    (gdb) n
    (gdb) p x
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
    context.getOrLoadDialect<mlir::arith::ArithDialect>();    // For constants

    mlir::registerBuiltinDialectTranslation( context );
    mlir::registerLLVMDialectTranslation( context );

/*
1:const char * str = "hi";
2:int main() { // this is a dummy program corresponding to the LLVM-IR produced by prototypes/simplest.cpp
3:    long x = 42;
4:    char * str_ptr = str;
5:    return 0;
6:}
*/
#define MODULE_LINE 1
#define main_LINE 2
#define x_LINE 3
#define str_ptr_LINE 4
#define return_LINE 5
#define THE_COLUMN_START 3
    mlir::OpBuilder builder( &context );
    auto moduleLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), MODULE_LINE, 1 );
    auto module = mlir::ModuleOp::create( moduleLoc );
    module->setAttr( "llvm.data_layout", builder.getStringAttr( dataLayoutStr ) );
    module->setAttr( "llvm.ident", builder.getStringAttr( "sillycompiler 0.0" ) );
    module->setAttr( "llvm.target_triple", builder.getStringAttr( targetTriple ) );

    // Create global string constant
    builder.setInsertionPointToStart( module.getBody() );
    auto i8Type = builder.getI8Type();
    auto arrayType = mlir::LLVM::LLVMArrayType::get( i8Type, 3 );    // "hi\00"
    llvm::SmallVector<char> stringData = { 'h', 'i', '\0' };
    auto denseAttr = mlir::DenseElementsAttr::get( mlir::RankedTensorType::get( { 3 }, i8Type ),
                                                   llvm::ArrayRef<char>( stringData ) );
    auto globalOp =
        builder.create<mlir::LLVM::GlobalOp>( moduleLoc, arrayType,
                                              /*isConstant=*/true, mlir::LLVM::Linkage::Private, "str_0", denseAttr );
    globalOp->setAttr( "unnamed_addr", builder.getUnitAttr() );

    // Create main function
    auto mainLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), main_LINE, 1 );
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType( {}, i32Type );
    auto func = builder.create<mlir::func::FuncOp>( mainLoc, "main", funcType );
    auto &block = *func.addEntryBlock();
    builder.setInsertionPointToStart( &block );

    // Debug info for the compilation unit and subprogram:
    auto fileAttr = mlir::LLVM::DIFileAttr::get( &context, "test.c", "." );
    auto distinctAttr = mlir::DistinctAttr::create( builder.getUnitAttr() );
    auto compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
        &context, distinctAttr, llvm::dwarf::DW_LANG_C, fileAttr, builder.getStringAttr( "testcompiler" ), false,
        mlir::LLVM::DIEmissionKind::Full, mlir::LLVM::DINameTableKind::Default );
    auto returnTypeAttr = mlir::LLVM::DIBasicTypeAttr::get( &context, (unsigned)llvm::dwarf::DW_TAG_base_type,
                                                            builder.getStringAttr( "int" ), TARGET_INT_SIZE_IN_BITS,
                                                            (unsigned)llvm::dwarf::DW_ATE_signed );
    llvm::SmallVector<mlir::LLVM::DITypeAttr, 1> typeArray;
    typeArray.push_back( returnTypeAttr );
    auto subprogramType = mlir::LLVM::DISubroutineTypeAttr::get( &context, 0, typeArray );
    auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
        &context, mlir::DistinctAttr::create( builder.getUnitAttr() ), compileUnitAttr, fileAttr,
        builder.getStringAttr( "main" ), builder.getStringAttr( "main" ), fileAttr, main_LINE, main_LINE,
        mlir::LLVM::DISubprogramFlags::Definition, subprogramType, llvm::ArrayRef<mlir::LLVM::DINodeAttr>{},
        llvm::ArrayRef<mlir::LLVM::DINodeAttr>{} );
    func->setAttr( "llvm.debug.subprogram", subprogramAttr );

    // Add (long) variable x
    auto varLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), x_LINE, THE_COLUMN_START );
    auto ptrType = mlir::LLVM::LLVMPointerType::get( &context );
    auto i64Type = builder.getI64Type();
    auto constOneI64 = builder.create<mlir::LLVM::ConstantOp>( varLoc, i64Type, builder.getI64IntegerAttr( 1 ) );
    auto allocaOp =
        builder.create<mlir::LLVM::AllocaOp>( varLoc, ptrType, i64Type, constOneI64, TARGET_LONG_SIZE_IN_BYTES );

    // Variable DI for x
    allocaOp->setAttr( "bindc_name", builder.getStringAttr( "x" ) );
    auto di_xType =
        mlir::LLVM::DIBasicTypeAttr::get( &context, llvm::dwarf::DW_TAG_base_type, builder.getStringAttr( "long" ),
                                          TARGET_LONG_SIZE_IN_BITS, llvm::dwarf::DW_ATE_signed );
    auto di_xVar =
        mlir::LLVM::DILocalVariableAttr::get( &context, subprogramAttr, builder.getStringAttr( "x" ), fileAttr, x_LINE,
                                              0, TARGET_LONG_SIZE_IN_BITS, di_xType, mlir::LLVM::DIFlags::Zero );
    builder.create<mlir::LLVM::DbgDeclareOp>( varLoc, allocaOp, di_xVar );

    // store 42 into x
    auto const42 = builder.create<mlir::LLVM::ConstantOp>( varLoc, i64Type, builder.getI64IntegerAttr( 42 ) );
    builder.create<mlir::LLVM::StoreOp>( varLoc, const42, allocaOp );

    // Reference the global string (e.g., store its address to a pointer)
    auto strPtrLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), str_ptr_LINE, THE_COLUMN_START );
    auto strPtrAlloca =
        builder.create<mlir::LLVM::AllocaOp>( strPtrLoc, ptrType, ptrType, constOneI64, TARGET_POINTER_SIZE_IN_BYTES );

    // Variable DI for str_ptr
    strPtrAlloca->setAttr( "bindc_name", builder.getStringAttr( "str_ptr" ) );
    auto charType =
        mlir::LLVM::DIBasicTypeAttr::get( &context, llvm::dwarf::DW_TAG_base_type, builder.getStringAttr( "char" ),
                                          TARGET_CHAR_SIZE_IN_BITS, llvm::dwarf::DW_ATE_signed_char );
    auto di_str_ptrType = mlir::LLVM::DIDerivedTypeAttr::get(
        &context, llvm::dwarf::DW_TAG_pointer_type, builder.getStringAttr( "char *" ), charType,
        TARGET_POINTER_SIZE_IN_BITS, TARGET_POINTER_SIZE_IN_BITS, 0, std::nullopt, mlir::LLVM::DINodeAttr() );
    auto di_str_ptrVar = mlir::LLVM::DILocalVariableAttr::get(
        &context, subprogramAttr, builder.getStringAttr( "str_ptr" ), fileAttr, str_ptr_LINE, 0,
        TARGET_POINTER_SIZE_IN_BITS, di_str_ptrType, mlir::LLVM::DIFlags::Zero );

    builder.create<mlir::LLVM::DbgDeclareOp>( strPtrLoc, strPtrAlloca, di_str_ptrVar );

    // store &global_string_literal into str_ptr
    auto globalPtr = builder.create<mlir::LLVM::AddressOfOp>( strPtrLoc, globalOp );
    builder.create<mlir::LLVM::StoreOp>( strPtrLoc, globalPtr, strPtrAlloca );

    // Return 0
    auto retLoc = mlir::FileLineColLoc::get( builder.getStringAttr( "test.c" ), return_LINE, THE_COLUMN_START );
    auto zero = builder.create<mlir::LLVM::ConstantOp>( retLoc, i32Type, builder.getI32IntegerAttr( 0 ) );
    builder.create<mlir::func::ReturnOp>( retLoc, mlir::ValueRange{ zero } );

    // The following fused location must be set before the final call to translateModuleToLLVMIR(), and is different
    // from all the rest of the normal MLIR loc() info.
    //
    // Without this, translateModuleToLLVMIR() strips out all the location info and doesn't convert the MLIR loc()'s to
    // !dbg statements
    //  (this was painful to figure out.)
    func->setLoc( builder.getFusedLoc( { mainLoc }, subprogramAttr ) );

    // Causes: already in an operation block:
    // module.push_back( func );

    // llvm::outs() << "Dump module before passes:\n";
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm().enableDebugInfo( true );
    // module.print( llvm::outs(), flags );

    // Run passes
    mlir::PassManager pm( &context );
    pm.addPass( mlir::createConvertFuncToLLVMPass() );
    if ( failed( pm.run( module ) ) )
    {
        llvm::errs() << "Failed to run conversion passes\n";
        return 1;
    }

    // llvm::errs() << "After lowering all but module (before translateModuleToLLVMIR):\n\n";
    // module.print( llvm::outs(), flags );

    // Translate to LLVM IR
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
