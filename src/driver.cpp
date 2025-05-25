/**
 * @file    driver.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   This is the compiler driver for the toy calculator compiler.
 *
 * @description
 *
 * This file orchestrates all the compiler actions:
 *
 * - command line options handling,
 * - runs the antlr4 parse tree listener (w/ MLIR builder),
 * - runs the LLVM-IR lowering pass, and
 * - runs the assembly printer.
 */
#include <assert.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DebugProgramInstruction.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <format>
#include <fstream>

#include "ToyDialect.h"
#include "ToyExceptions.h"
#include "ToyLexer.h"
#include "ToyPasses.h"
#include "driver.h"
#include "lowering.h"
#include "parser.h"

#define DEBUG_TYPE "toy-driver"

// Define a category for Toy Calculator options
static llvm::cl::OptionCategory ToyCategory( "Toy Calculator Options" );

// Command-line option for input file
static llvm::cl::opt<std::string> inputFilename( llvm::cl::Positional, llvm::cl::desc( "<input file>" ),
                                                 llvm::cl::init( "-" ), llvm::cl::value_desc( "filename" ),
                                                 llvm::cl::cat( ToyCategory ), llvm::cl::NotHidden );

static llvm::cl::opt<bool> debugInfo( "g",
                                      llvm::cl::desc( "Enable location output in MLIR, and dwarf metadata "
                                                      "creation in the lowered LLVM IR)" ),
                                      llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

static llvm::cl::opt<bool> compileOnly( "c",
                                      llvm::cl::desc( "Compile only and don't link." ),
                                      llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

static llvm::cl::opt<std::string> outDir(
    "output-directory", llvm::cl::desc( "Output directory for generated files (e.g., .mlir, .ll, .o)" ),
    llvm::cl::value_desc( "directory" ), llvm::cl::init( "" ), llvm::cl::cat( ToyCategory ) );

static llvm::cl::opt<bool> toStdout( "stdout", llvm::cl::desc( "LLVM and MLIR on stdout instead of to a file" ),
                                     llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

// Add command-line option for MLIR emission
static llvm::cl::opt<bool> emitMLIR( "emit-mlir", llvm::cl::desc( "Emit MLIR IR" ), llvm::cl::init( false ),
                                     llvm::cl::cat( ToyCategory ) );

// Add command-line option for LLVM IR emission
static llvm::cl::opt<bool> emitLLVM( "emit-llvm", llvm::cl::desc( "Emit LLVM IR" ), llvm::cl::init( false ),
                                     llvm::cl::cat( ToyCategory ) );

// Add command-line option for object file emission
static llvm::cl::opt<bool> noEmitObject( "no-emit-object", llvm::cl::desc( "Skip emit object file (.o)" ),
                                         llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

// Noisy debugging output
static llvm::cl::opt<bool> llvmDEBUG( "debug-llvm", llvm::cl::desc( "Include MLIR dump, and turn off multithreading" ),
                                      llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

enum class OptLevel : int
{
    O0,
    O1,
    O2,
    O3
};

static llvm::cl::opt<OptLevel> optLevel( "O", llvm::cl::desc( "Optimization level" ),
                                         llvm::cl::values( clEnumValN( OptLevel::O0, "0", "No optimization" ),
                                                           clEnumValN( OptLevel::O1, "1", "Light optimization" ),
                                                           clEnumValN( OptLevel::O2, "2", "Moderate optimization" ),
                                                           clEnumValN( OptLevel::O3, "3", "Aggressive optimization" ) ),
                                         llvm::cl::init( OptLevel::O0 ), llvm::cl::cat( ToyCategory ) );

enum class return_codes : int
{
    success,
    cannot_open_file,
    semantic_error,
    unknown_error
};

static
void invokeLinker( const char* argv0, llvm::SmallString<128> & exePath, llvm::SmallString<128> & objectPath );

using namespace toy;

int main( int argc, char** argv )
{
    // Initialize LLVM targets for code generation
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Calculator compiler\n" );

    std::ifstream inputStream;
    std::string filename = inputFilename;
    if ( filename != "-" )
    {
        inputStream.open( filename );
        if ( !inputStream.is_open() )
        {
            llvm::errs() << std::format( "Error: Cannot open file {}\n", filename );
            return (int)return_codes::cannot_open_file;
        }
    }
    else
    {
        filename = "<stdin>.toy";
        inputStream.basic_ios<char>::rdbuf( std::cin.rdbuf() );
    }

    try
    {
        MLIRListener listener( filename );

        antlr4::ANTLRInputStream antlrInput( inputStream );
        ToyLexer lexer( &antlrInput );
        antlr4::CommonTokenStream tokens( &lexer );
        ToyParser parser( &tokens );

        antlr4::tree::ParseTree* tree = parser.startRule();
        antlr4::tree::ParseTreeWalker::DEFAULT.walk( &listener, tree );

        // For now, always dump the original MLIR unconditionally, even if we
        // are doing the LLVM IR lowering pass:
        mlir::OpPrintingFlags flags;
        if ( debugInfo )
        {
            flags.printGenericOpForm().enableDebugInfo( true );
        }
        llvm::StringRef stem = llvm::sys::path::stem( filename );
        if ( stem.empty() )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          "Invalid filename: empty stem: '" + filename + "'" );
        }
        llvm::StringRef dirname = llvm::sys::path::parent_path( filename );
        llvm::SmallString<128> dirWithStem;

        // Create output directory if specified
        if ( !outDir.empty() )
        {
            std::error_code EC = llvm::sys::fs::create_directories( outDir );
            if ( EC )
            {
                throw exception_with_context( __FILE__, __LINE__, __func__,
                                              "Failed to create output directory: " + EC.message() );
            }
            dirWithStem = outDir;
            llvm::sys::path::append( dirWithStem, stem );
        }
        else if ( dirname != "" )
        {
            dirWithStem = dirname;
            llvm::sys::path::append( dirWithStem, stem );
        }
        else
        {
            dirWithStem = stem;
        }

        if ( emitMLIR )
        {
            if ( toStdout )
            {
                listener.getModule().print( llvm::outs(), flags );
            }
            else
            {
                llvm::SmallString<128> path = dirWithStem;
                path += ".mlir";
                std::error_code EC;
                llvm::raw_fd_ostream out( path.str(), EC, llvm::sys::fs::OF_Text );
                if ( EC )
                {
                    throw exception_with_context( __FILE__, __LINE__, __func__,
                                                  "Failed to open file: " + EC.message() );
                }
                listener.getModule().print( out, flags );
            }
        }

        auto module = listener.getModule();
        auto context = module.getContext();

        // Register dialect translations
        mlir::registerLLVMDialectTranslation( *context );
        mlir::registerBuiltinDialectTranslation( *context );

        if ( llvmDEBUG )
        {
            context->disableMultithreading( true );
        }

        // Set up pass manager for lowering
        mlir::PassManager pm( context );
        if ( llvmDEBUG )
        {
            pm.enableIRPrinting();
        }

        driverState st;
        st.isOptimized = optLevel != OptLevel::O0 ? true : false;
        st.filename = filename;

        pm.addPass( mlir::createToyToLLVMLoweringPass( &st ) );
        pm.addPass( mlir::createConvertSCFToCFPass() );
        pm.addPass( mlir::createConvertFuncToLLVMPass() );
        pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
        pm.addPass( mlir::createConvertControlFlowToLLVMPass() );

        if ( llvm::failed( pm.run( module ) ) )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__, "LLVM lowering failed" );
        }

        if ( toStdout )
        {
            llvm::outs() << "Before module lowering:\n";
            module.print( llvm::outs(), flags );
        }

        // Export to LLVM IR
        llvm::LLVMContext llvmContext;

        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR( module, llvmContext, filename );

        if ( !llvmModule )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__, "Failed to translate to LLVM IR" );
        }

        auto emitObject = !noEmitObject;
        if ( emitLLVM || emitObject )
        {
            if ( emitLLVM )
            {
                // Verify the module to ensure debug info is valid
                if ( llvm::verifyModule( *llvmModule, &llvm::errs() ) )
                {
                    throw exception_with_context( __FILE__, __LINE__, __func__, "Invalid LLVM IR module" );
                }

                if ( toStdout )
                {
                    llvmModule->print( llvm::outs(), nullptr, debugInfo /* print debug info */ );
                }
                else
                {
                    llvm::SmallString<128> path = dirWithStem;
                    path += ".ll";
                    std::error_code EC;
                    llvm::raw_fd_ostream out( path.str(), EC, llvm::sys::fs::OF_Text );
                    if ( EC )
                    {
                        throw exception_with_context( __FILE__, __LINE__, __func__,
                                                      "Failed to open file: " + EC.message() );
                    }

                    llvmModule->print( out, nullptr, debugInfo /* print debug info */ );
                }
            }

            if ( emitObject )
            {
                // Set target triple
                std::string targetTriple = llvm::sys::getProcessTriple();
                llvmModule->setTargetTriple( targetTriple );

                // Lookup the target
                std::string error;
                const llvm::Target* target = llvm::TargetRegistry::lookupTarget( targetTriple, error );
                if ( !target )
                {
                    throw exception_with_context( __FILE__, __LINE__, __func__, "Failed to find target: " + error );
                }

                // Create the target machine
                std::unique_ptr<llvm::TargetMachine> targetMachine(
                    target->createTargetMachine( targetTriple, "generic", "", llvm::TargetOptions(), std::nullopt ) );
                if ( !targetMachine )
                {
                    throw exception_with_context( __FILE__, __LINE__, __func__, "Failed to create target machine" );
                }

                // Optimize the module (optional)
                llvm::PassBuilder passBuilder( targetMachine.get() );
                llvm::LoopAnalysisManager LAM;
                llvm::FunctionAnalysisManager FAM;
                llvm::CGSCCAnalysisManager CGAM;
                llvm::ModuleAnalysisManager MAM;

                passBuilder.registerModuleAnalyses( MAM );
                passBuilder.registerCGSCCAnalyses( CGAM );
                passBuilder.registerFunctionAnalyses( FAM );
                passBuilder.registerLoopAnalyses( LAM );
                passBuilder.crossRegisterProxies( LAM, FAM, CGAM, MAM );

                llvm::OptimizationLevel opt;
                switch ( optLevel )
                {
                    case OptLevel::O0:
                        opt = llvm::OptimizationLevel::O0;
                        break;
                    case OptLevel::O1:
                        opt = llvm::OptimizationLevel::O1;
                        break;
                    case OptLevel::O2:
                        opt = llvm::OptimizationLevel::O2;
                        break;
                    case OptLevel::O3:
                        opt = llvm::OptimizationLevel::O3;
                        break;
                }
                llvm::ModulePassManager MPM = passBuilder.buildPerModuleDefaultPipeline( opt );

                MPM.run( *llvmModule, MAM );

                // Emit object file
                llvm::SmallString<128> outputFilename( dirWithStem );
                outputFilename += ".o";
                std::error_code EC;
                llvm::raw_fd_ostream dest( outputFilename.str(), EC, llvm::sys::fs::OF_None );
                if ( EC )
                {
                    throw exception_with_context( __FILE__, __LINE__, __func__,
                                                  "Failed to open output file: " + EC.message() );
                }

                llvmModule->setDataLayout( targetMachine->createDataLayout() );
                llvm::legacy::PassManager codegenPM;
                if ( targetMachine->addPassesToEmitFile( codegenPM, dest, nullptr, llvm::CodeGenFileType::ObjectFile ) )
                {
                    throw exception_with_context( __FILE__, __LINE__, __func__,
                                                  "TargetMachine can't emit an object file" );
                }

                codegenPM.run( *llvmModule );
                dest.close();

                LLVM_DEBUG( { llvm::outs() << "Generated object file: " << outputFilename << '\n'; } );

                if ( compileOnly == false )
                {
                    invokeLinker( argv[0], dirWithStem, outputFilename );
                }
            }
        }
    }
    catch ( const std::exception& e )
    {
        llvm::errs() << std::format( "FATAL ERROR: {}\n", e.what() );
        return (int)return_codes::unknown_error;
    }

    return (int)return_codes::success;
}

void invokeLinker( const char* argv0, llvm::SmallString<128> & exePath, llvm::SmallString<128> & objectPath )
{
    // Get the driver path
    std::string driver = llvm::sys::fs::getMainExecutable( argv0, (void*)&main );
    llvm::StringRef driverPath = llvm::sys::path::parent_path( driver );
    LLVM_DEBUG( { llvm::outs() << "Compiler driver path: " << driverPath << '\n'; } );

    // Find the linker (gcc)
    auto linker = "gcc";
    auto linkerPath = llvm::sys::findProgramByName( linker );
    if ( !linkerPath )
    {
        std::error_code ec = linkerPath.getError();

        throw exception_with_context( __FILE__, __LINE__, __func__,
                            std::format( "Error finding path for linker '{}': {}\n", linker, ec.message() ) );
    }
    LLVM_DEBUG( { llvm::outs() << "Linker path: " << linkerPath.get() << '\n'; } );

    // Construct the -Wl,-rpath argument
    llvm::SmallString<128> rpathOption;
    rpathOption.assign( "-Wl,-rpath," );
    rpathOption.append( driverPath );

    // Create argv for ExecuteAndWait
    llvm::SmallVector<llvm::StringRef, 16> argv;
    argv.push_back( linkerPath.get() );
    argv.push_back( "-g" );
    argv.push_back( "-o" );
    argv.push_back( exePath );
    argv.push_back( objectPath );
    argv.push_back( "-L" );
    argv.push_back( driverPath );
    argv.push_back( "-l" );
    argv.push_back( "toy_runtime" );
    argv.push_back( rpathOption );

    // Execute the linker
    std::string errMsg;
    int result = llvm::sys::ExecuteAndWait( linkerPath.get(), argv, std::nullopt, {}, 0, 0, &errMsg );
    if ( result != 0 )
    {
        throw exception_with_context( __FILE__, __LINE__, __func__,
                            std::format( "Linker failed with exit code: {}, rc = {}\n", errMsg, result ) );
    }
}

// vim: et ts=4 sw=4
