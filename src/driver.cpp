/// @file    driver.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This is the compiler driver for the toy (originally "calculator") compiler.
///
/// @description
///
/// This file orchestrates all the compiler actions:
///
/// - command line options handling,
/// - runs the antlr4 parse tree listener (w/ MLIR builder),
/// - runs the LLVM-IR lowering pass, and
/// - runs the assembly printer.
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

#include "ToyDialect.hpp"
#include "ToyExceptions.hpp"
#include "ToyLexer.h"
#include "ToyPasses.hpp"
#include "driver.hpp"
#include "lowering.hpp"
#include "parser.hpp"

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

static llvm::cl::opt<bool> compileOnly( "c", llvm::cl::desc( "Compile only and don't link." ), llvm::cl::init( false ),
                                        llvm::cl::cat( ToyCategory ) );

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

static void invokeLinker( const char* argv0, llvm::SmallString<128>& exePath, llvm::SmallString<128>& objectPath );

static void writeLL( std::unique_ptr<llvm::Module>& llvmModule, llvm::SmallString<128>& dirWithStem )
{
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
            throw toy::ExceptionWithContext( __FILE__, __LINE__, __func__, "Failed to open file: " + EC.message() );
        }

        llvmModule->print( out, nullptr, debugInfo /* print debug info */ );
    }
}

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

        // Remove default error listener and add MLIRListener for errors
        parser.removeErrorListeners();
        parser.addErrorListener( &listener );

        antlr4::tree::ParseTree* tree = parser.startRule();
        antlr4::tree::ParseTreeWalker::DEFAULT.walk( &listener, tree );

        // For now, always dump the original MLIR unconditionally, even if we
        // are doing the LLVM IR lowering pass:
        mlir::OpPrintingFlags flags;
        if ( debugInfo )
        {
            flags.enableDebugInfo( true );
            // flags.printGenericOpForm(); // Why did I do this?  If I have an assemblyFormat, I'd like it to show up.
        }
        llvm::StringRef stem = llvm::sys::path::stem( filename );
        if ( stem.empty() )
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
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
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
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
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Failed to open file: " + EC.message() );
                }
                listener.getModule().print( out, flags );
            }
        }

        mlir::ModuleOp& module = listener.getModule();
        mlir::MLIRContext* context = module.getContext();

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
        st.wantDebug = debugInfo;
        st.filename = filename;

        LLVM_DEBUG( {
            llvm::errs() << "IR before stage I lowering:\n";
            module->dump();
        } );

        pm.addPass( mlir::createToyToLLVMLoweringPass( &st ) );
        pm.addPass( mlir::createSCFToControlFlowPass() );
        pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
        pm.addPass( mlir::createConvertControlFlowToLLVMPass() );

        if ( llvm::failed( pm.run( module ) ) )
        {
            llvm::errs() << "IR after stage I lowering failure:\n";
            module->dump();
            throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Stage I LLVM lowering failed" );
        }

        mlir::PassManager pm2( context );
        if ( llvmDEBUG )
        {
            pm2.enableIRPrinting();
        }

        pm2.addPass( mlir::createConvertFuncToLLVMPass() );

        if ( llvm::failed( pm2.run( module ) ) )
        {
            llvm::errs() << "IR after stage II lowering failure:\n";
            module->dump();
            throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Stage II LLVM lowering failed" );
        }

        if ( toStdout )
        {
            llvm::outs() << "Before module lowering:\n";
            module.print( llvm::outs(), flags );
        }

        // The module should now contain mostly LLVM-IR instructions, with the exception of the top level module,
        // and the MLIR style loc() references.  Those last two MLIR artifacts will be convered to LLVM-IR
        // now, also producing !DILocation's for all the loc()s.
        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR( module, llvmContext, filename );

        if ( !llvmModule )
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Failed to translate to LLVM IR" );
        }

        bool emitObject = !noEmitObject;
        if ( emitLLVM || emitObject )
        {
            // Verify the module to ensure debug info is valid
            if ( llvm::verifyModule( *llvmModule, &llvm::errs() ) )
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Invalid LLVM IR module" );
            }

            // Dump the pre-optimized LL if we aren't creating a .o
            if ( emitLLVM && !emitObject )
            {
                writeLL( llvmModule, dirWithStem );
            }

            if ( emitObject )
            {
                std::string targetTripleStr = llvm::sys::getProcessTriple();
                llvm::Triple targetTriple( targetTripleStr );
                llvmModule->setTargetTriple( targetTriple );

                // Lookup the target
                std::string error;
                const llvm::Target* target = llvm::TargetRegistry::lookupTarget( targetTriple, error );
                if ( !target )
                {
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Failed to find target: " + error );
                }

                // Create the target machine
                std::unique_ptr<llvm::TargetMachine> targetMachine(
                    target->createTargetMachine( targetTriple, "generic", "", llvm::TargetOptions(), std::nullopt ) );
                if ( !targetMachine )
                {
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__, "Failed to create target machine" );
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
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                                "Failed to open output file: " + EC.message() );
                }

                llvmModule->setDataLayout( targetMachine->createDataLayout() );
                llvm::legacy::PassManager codegenPM;
                if ( targetMachine->addPassesToEmitFile( codegenPM, dest, nullptr, llvm::CodeGenFileType::ObjectFile ) )
                {
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                                "TargetMachine can't emit an object file" );
                }

                codegenPM.run( *llvmModule );
                dest.close();

                LLVM_DEBUG( { llvm::outs() << "Generated object file: " << outputFilename << '\n'; } );

                writeLL( llvmModule, dirWithStem );

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

void invokeLinker( const char* argv0, llvm::SmallString<128>& exePath, llvm::SmallString<128>& objectPath )
{
    // Get the driver path
    std::string driver = llvm::sys::fs::getMainExecutable( argv0, (void*)&main );
    llvm::StringRef driverPath = llvm::sys::path::parent_path( driver );
    LLVM_DEBUG( { llvm::outs() << "Compiler driver path: " << driverPath << '\n'; } );

    // Find the linker (gcc)
    const char* linker = "gcc";
    llvm::ErrorOr<std::string> linkerPath = llvm::sys::findProgramByName( linker );
    if ( !linkerPath )
    {
        std::error_code ec = linkerPath.getError();

        throw ExceptionWithContext( __FILE__, __LINE__, __func__,
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
        throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                    std::format( "Linker failed with exit code: {}, rc = {}\n", errMsg, result ) );
    }
}

// vim: et ts=4 sw=4
