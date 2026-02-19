/// @file    driver.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This is the compiler driver for the silly compiler (originally "toy calculator").
///
/// This file orchestrates all the compiler actions:
///
/// - command line options handling,
/// - runs the antlr4 parse tree listener (w/ MLIR builder),
/// - runs the LLVM-IR lowering pass, and
/// - runs the assembly printer.
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
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <format>
#include <fstream>

#include "SillyDialect.hpp"
#include "SillyLexer.h"
#include "SillyPasses.hpp"
#include "DriverState.hpp"
#include "createSillyToLLVMLoweringPass.hpp"
#include "ParseListener.hpp"
#include "DialectContext.hpp"

/// --debug- class for the driver
#define DEBUG_TYPE "silly-driver"

/// Allowed optimization levels
enum class OptLevel : int
{
    O0,
    O1,
    O2,
    O3
};

/// The numeric return codes for the silly driver
enum class ReturnCodes : int
{
    success,
    badExtensionError,
    directoryError,
    filenameParseError,
    linkError,
    loweringError,
    openError,
    parseError,
    verifyError,
};

/// Supported source code file extensions.
enum class InputType
{
    Silly,     // .silly or other source
    MLIR,      // .mlir
    OBJECT,    // .o
    Unknown
};

// Define a category for silly compiler options
static llvm::cl::OptionCategory SillyCategory( "Silly Compiler Options" );

//--------------------------------
// Options related to output files
static llvm::cl::opt<std::string> inputFilename( llvm::cl::Positional, llvm::cl::desc( "<input file>" ),
                                                 llvm::cl::init( "-" ), llvm::cl::value_desc( "filename" ),
                                                 llvm::cl::cat( SillyCategory ), llvm::cl::NotHidden );

static llvm::cl::opt<bool> compileOnly( "c", llvm::cl::desc( "Compile only and don't link." ), llvm::cl::init( false ),
                                        llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<std::string> outDir(
    "output-directory", llvm::cl::desc( "Output directory for generated files (e.g., .mlir, .ll, .o)" ),
    llvm::cl::value_desc( "directory" ), llvm::cl::init( "" ), llvm::cl::cat( SillyCategory ) );

// Add command-line option for MLIR emission
static llvm::cl::opt<bool> emitMLIR( "emit-mlir", llvm::cl::desc( "Emit MLIR IR" ), llvm::cl::init( false ),
                                     llvm::cl::cat( SillyCategory ) );

// Add command-line option for LLVM IR emission
static llvm::cl::opt<bool> emitLLVM( "emit-llvm", llvm::cl::desc( "Emit LLVM IR" ), llvm::cl::init( false ),
                                     llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> toStdout( "stdout", llvm::cl::desc( "LLVM and MLIR on stdout instead of to a file" ),
                                     llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

//--------------------------------
// Options that change code generation:
static llvm::cl::opt<int> initFillValue( "init-fill", llvm::cl::desc( "Initializer fill value." ), llvm::cl::init( 0 ),
                                         llvm::cl::ValueRequired, llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<OptLevel> optLevel( "O", llvm::cl::desc( "Optimization level" ),
                                         llvm::cl::values( clEnumValN( OptLevel::O0, "0", "No optimization" ),
                                                           clEnumValN( OptLevel::O1, "1", "Light optimization" ),
                                                           clEnumValN( OptLevel::O2, "2", "Moderate optimization" ),
                                                           clEnumValN( OptLevel::O3, "3", "Aggressive optimization" ) ),
                                         llvm::cl::init( OptLevel::O0 ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> noAbortPath( "no-abort-path", llvm::cl::desc( "Specify to omit include source path in ABORT" ),
                                         llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> debugInfo( "g",
                                      llvm::cl::desc( "Enable location output in MLIR, and dwarf metadata "
                                                      "creation in the lowered LLVM IR)" ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

//--------------------------------
// Diagnostic options:
static llvm::cl::opt<bool> verboseLink( "verbose-link", llvm::cl::desc( "Display the link command line on stderr" ),
                                        llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

// Noisy debugging output (this is different than --debug which is intercepted by llvm itself)
static llvm::cl::opt<bool> llvmDEBUG( "debug-llvm", llvm::cl::desc( "Include MLIR dump, and turn off multithreading" ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> noColorErrors( "no-color-errors", llvm::cl::desc( "Disable color error messages" ),
                                          llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

//--------------------------------
static InputType getInputType( llvm::StringRef filename )
{
    llvm::StringRef ext = llvm::sys::path::extension( filename );

    if ( ext == ".mlir" || ext == ".sir" )
    {
        return InputType::MLIR;
    }

    if ( ext == ".silly" )
    {
        return InputType::Silly;
    }

    if ( ext == ".o" )
    {
        return InputType::OBJECT;
    }

    return InputType::Unknown;
}

static void llvmInitialization( int argc, char** argv )
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
}

static mlir::ModuleOp runParserAndBuilder( silly::ParseListener& listener, silly::DriverState& st,
                                           std::ifstream& inputStream )
{
    antlr4::ANTLRInputStream antlrInput( inputStream );
    SillyLexer lexer( &antlrInput );
    antlr4::CommonTokenStream tokens( &lexer );
    SillyParser parser( &tokens );

    // Remove default error listener and add ParseListener for errors
    parser.removeErrorListeners();
    parser.addErrorListener( &listener );

    antlr4::tree::ParseTree* tree = parser.startRule();
    antlr4::tree::ParseTreeWalker::DEFAULT.walk( &listener, tree );

    return listener.getModule();
}

/// Create the directory named in --output-directory
///
/// @param theDirectory [out] That output directory, or the directory part of the filename path (if specified), or an empty string.
///
static void makeOutputDirectory( const std::string& filename, llvm::SmallString<128>& theDirectory )
{
    llvm::StringRef dirname = llvm::sys::path::parent_path( filename );
    // Create output directory if specified
    if ( !outDir.empty() )
    {
        std::error_code EC = llvm::sys::fs::create_directories( outDir );
        if ( EC )
        {
            std::string dir = outDir;
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create output directory '{}': {}\n", dir,
                                         EC.message() );
            std::exit( (int)ReturnCodes::directoryError );
        }

        theDirectory = outDir;
    }
    else if ( dirname != "" )
    {
        theDirectory = dirname;
    }
}

static void serializeModuleMLIR( mlir::ModuleOp mod, mlir::OpPrintingFlags flags,
                                 const llvm::SmallString<128>& dirWithStem )
{
    if ( emitMLIR )
    {
        if ( toStdout )
        {
            mod.print( llvm::outs(), flags );
        }
        else
        {
            llvm::SmallString<128> path = dirWithStem;
            path += ".mlir";
            std::error_code EC;
            llvm::raw_fd_ostream out( path.str(), EC, llvm::sys::fs::OF_Text );
            if ( EC )
            {
                llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file {}: {}\n", std::string( path ),
                                             EC.message() );
                std::exit( (int)ReturnCodes::openError );
            }
            mod.print( out, flags );
        }
    }
}

static void writeLL( std::unique_ptr<llvm::Module>& llvmModule, const llvm::SmallString<128>& dirWithStem )
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
            // FIXME: probably want llvm::formatv here and elsewhere to avoid the std::string casting hack (assuming
            // it knows how to deal with StringRef)
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open file '{}': {}\n", std::string( path ),
                                         EC.message() );
            std::exit( (int)ReturnCodes::openError );
        }

        llvmModule->print( out, nullptr, debugInfo /* print debug info */ );
    }
}

static void showLinkCommand( const std::string& linker, llvm::SmallVector<llvm::StringRef, 16>& args )
{
    llvm::errs() << "# " << linker;

    for ( const auto& a : args )
    {
        llvm::errs() << a << ' ';
    }

    llvm::errs() << '\n';
}

static void invokeLinker( const char* argv0, const llvm::SmallString<128>& exePath,
                          const llvm::SmallString<128>& objectPath, void* mainSymbol, silly::DriverState& st )
{
    // Get the driver path
    std::string driver = llvm::sys::fs::getMainExecutable( argv0, mainSymbol );
    llvm::StringRef driverPath = llvm::sys::path::parent_path( driver );
    LLVM_DEBUG( { llvm::outs() << "Compiler driver path: " << driverPath << '\n'; } );

    // Find the linker (gcc)
    const char* linker = "gcc";
    llvm::ErrorOr<std::string> linkerPath = llvm::sys::findProgramByName( linker );
    if ( !linkerPath )
    {
        std::error_code EC = linkerPath.getError();

        llvm::errs() << std::format( COMPILER_NAME ": error: Error finding path for linker '{}': {}\n", linker,
                                     EC.message() );
        std::exit( (int)ReturnCodes::filenameParseError );
    }
    LLVM_DEBUG( { llvm::outs() << "Linker path: " << linkerPath.get() << '\n'; } );

    // Construct paths that need to persist
    llvm::SmallString<128> libPath;
    libPath.assign( driverPath );
    libPath.append( "/../../lib" );

    llvm::SmallString<128> rpathOption;
    rpathOption.assign( "-Wl,-rpath," );
    rpathOption.append( driverPath );
    rpathOption.append( "/../../lib" );

    // Create args for ExecuteAndWait
    llvm::SmallVector<llvm::StringRef, 16> linkerArgs;
    linkerArgs.push_back( linkerPath.get() );
    linkerArgs.push_back( "-g" );
    linkerArgs.push_back( "-o" );
    linkerArgs.push_back( exePath );
    linkerArgs.push_back( objectPath );
    linkerArgs.push_back( "-L" );
    linkerArgs.push_back( libPath );
    linkerArgs.push_back( "-l" );
    linkerArgs.push_back( "silly_runtime" );
    linkerArgs.push_back( rpathOption );
    if ( st.needsMathLib )
    {
        linkerArgs.push_back( "-lm" );
    }

    if ( verboseLink == true )
    {
        showLinkCommand( linkerPath.get(), linkerArgs );
    }

    // Execute the linker
    std::string errMsg;
    int result = llvm::sys::ExecuteAndWait( linkerPath.get(), linkerArgs, std::nullopt, {}, 0, 0, &errMsg );
    if ( result != 0 )
    {
        if ( verboseLink == false )    // already showed this
        {
            showLinkCommand( linkerPath.get(), linkerArgs );
        }

        llvm::errs() << std::format( COMPILER_NAME ": error: Linker failed with exit code: {}, rc = {}\n", errMsg,
                                     result );
        std::exit( (int)ReturnCodes::linkError );
    }
}

static void assembleAndLink( const llvm::SmallString<128>& dirWithStem, const char* argv0, void* mainSymbol,
                             std::unique_ptr<llvm::Module>& llvmModule, silly::DriverState& st )
{
    std::string targetTripleStr = llvm::sys::getProcessTriple();
    llvm::Triple targetTriple( targetTripleStr );
    llvmModule->setTargetTriple( targetTriple );

    // Lookup the target
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget( targetTriple, error );
    if ( !target )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to find target: {}\n", error );
        std::exit( (int)ReturnCodes::loweringError );
    }

    // Create the target machine
    std::unique_ptr<llvm::TargetMachine> targetMachine(
        target->createTargetMachine( targetTriple, "generic", "", llvm::TargetOptions(), std::nullopt ) );
    if ( !targetMachine )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create target machine\n" );
        std::exit( (int)ReturnCodes::loweringError );
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

    // Emit object file.  FIXME: we should use a temporary path if compileOnly was not set.
    llvm::SmallString<128> outputFilename( dirWithStem );
    outputFilename += ".o";
    std::error_code EC;
    llvm::raw_fd_ostream dest( outputFilename.str(), EC, llvm::sys::fs::OF_None );
    if ( EC )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open output file '{}': {}\n",
                                     std::string( outputFilename ), EC.message() );
        std::exit( (int)ReturnCodes::openError );
    }

    llvmModule->setDataLayout( targetMachine->createDataLayout() );
    llvm::legacy::PassManager codegenPM;
    if ( targetMachine->addPassesToEmitFile( codegenPM, dest, nullptr, llvm::CodeGenFileType::ObjectFile ) )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: TargetMachine can't emit an object file\n" );
        std::exit( (int)ReturnCodes::loweringError );
    }

    codegenPM.run( *llvmModule );
    dest.close();

    LLVM_DEBUG( { llvm::outs() << "Generated object file: " << outputFilename << '\n'; } );

    if ( compileOnly == false )
    {
        invokeLinker( argv0, dirWithStem, outputFilename, mainSymbol, st );
    }
}

static void lowerAssembleAndLinkModule( mlir::ModuleOp mod, const llvm::SmallString<128>& dirWithStem,
                                        silly::DriverState& st, mlir::OpPrintingFlags flags, const char* argv0,
                                        void* mainSymbol )
{
    mlir::MLIRContext* context = mod.getContext();

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

    LLVM_DEBUG( {
        llvm::errs() << "IR before stage I lowering:\n";
        mod->dump();
    } );

    pm.addPass( mlir::createSillyToLLVMLoweringPass( &st ) );
    pm.addPass( mlir::createSCFToControlFlowPass() );
    pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
    pm.addPass( mlir::createConvertControlFlowToLLVMPass() );

    if ( llvm::failed( pm.run( mod ) ) )
    {
        llvm::errs() << "IR after stage I lowering failure:\n";
        mod->dump();
        llvm::errs() << COMPILER_NAME ": error: Stage I LLVM lowering failed\n";
        std::exit( (int)ReturnCodes::loweringError );
    }

    mlir::PassManager pm2( context );
    if ( llvmDEBUG )
    {
        pm2.enableIRPrinting();
    }

    pm2.addPass( mlir::createConvertFuncToLLVMPass() );

    if ( llvm::failed( pm2.run( mod ) ) )
    {
        llvm::errs() << "IR after stage II lowering failure:\n";
        mod->dump();
        llvm::errs() << COMPILER_NAME ": error: Stage II LLVM lowering failed\n";
        std::exit( (int)ReturnCodes::loweringError );
    }

    if ( toStdout )
    {
        llvm::outs() << "Before module lowering:\n";
        mod.print( llvm::outs(), flags );
    }

    // The module should now contain mostly LLVM-IR instructions, with the exception of the top level module,
    // and the MLIR style loc() references.  Those last two MLIR artifacts will be convered to LLVM-IR
    // now, also producing !DILocation's for all the loc()s.
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR( mod, llvmContext, st.filename );

    if ( !llvmModule )
    {
        llvm::errs() << COMPILER_NAME ": error: Failed to translate to LLVM IR\n";
        std::exit( (int)ReturnCodes::loweringError );
    }

    // Verify the module to ensure debug info is valid
    if ( llvm::verifyModule( *llvmModule, &llvm::errs() ) )
    {
        llvm::errs() << COMPILER_NAME ": error: Invalid LLVM IR module\n";
        std::exit( (int)ReturnCodes::loweringError );
    }

    // Dump the pre-optimized LL if we aren't creating a .o
    if ( emitLLVM )
    {
        writeLL( llvmModule, dirWithStem );
    }

    assembleAndLink( dirWithStem, argv0, mainSymbol, llvmModule, st );
}

static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRFile( const std::string& filename, mlir::MLIRContext* context )
{
    auto fileOrErr = llvm::MemoryBuffer::getFile( filename );
    if ( std::error_code EC = fileOrErr.getError() )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file '{}': {}\n", filename, EC.message() );
        std::exit( (int)ReturnCodes::openError );
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer( std::move( *fileOrErr ), llvm::SMLoc{} );

    auto mod = mlir::parseSourceFile<mlir::ModuleOp>( sourceMgr, context );
    if ( !mod )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to parse MLIR file '{}'\n", filename );
        std::exit( (int)ReturnCodes::parseError );
    }

    return mod;
}

int main( int argc, char** argv )
{
    llvmInitialization( argc, argv );

    silly::DriverState st;
    st.isOptimized = optLevel != OptLevel::O0 ? true : false;
    st.fillValue = (uint8_t)initFillValue;
    st.wantDebug = debugInfo;
    st.colorErrors = !noColorErrors;
    st.abortOmitPath = noAbortPath;
    st.filename = inputFilename;

    InputType ity = getInputType( st.filename );
    if ( ity == InputType::Unknown )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: filename {} extension is none of .silly, .mlir, or .o\n",
                                     st.filename );
        std::exit( (int)ReturnCodes::badExtensionError );
    }

    // once this goes out of scope, the module is toast and can't be referenced further.
    silly::DialectContext dialectLoader;

    mlir::OpPrintingFlags flags;
    if ( debugInfo )
    {
        flags.enableDebugInfo( true );
    }

    silly::ParseListener listener( st, &dialectLoader.context );

    mlir::ModuleOp mod;
    mlir::OwningOpRef<mlir::ModuleOp> rmod;

    llvm::SmallString<128> dirWithStem;

    makeOutputDirectory( st.filename, dirWithStem );

    llvm::StringRef stem = llvm::sys::path::stem( st.filename ); // foo/bar.silly -> stem: is just bar, not foo/bar
    if ( stem.empty() )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Invalid filename '{}', empty stem\n", st.filename );
        std::exit( (int)ReturnCodes::filenameParseError );
    }

    if ( dirWithStem.empty() )
    {
        dirWithStem = stem;
    }
    else
    {
        llvm::sys::path::append( dirWithStem, stem );
    }

    if ( ity == InputType::Silly )
    {
        std::ifstream inputStream;
        inputStream.open( st.filename );
        if ( !inputStream.is_open() )
        {
            llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file {}\n", st.filename );
            std::exit( (int)ReturnCodes::openError );
        }

        mod = runParserAndBuilder( listener, st, inputStream );
        if ( !mod )
        {
            // should have already emitted diagnostics.
            std::exit( (int)ReturnCodes::parseError );
        }
    }
    else if ( ity == InputType::MLIR )
    {
        rmod = parseMLIRFile( st.filename, &dialectLoader.context );
        mod = rmod.get();
    }
    else    // .o
    {
        // dirWithStem is actually the name of the exe to link at this point and may or may
        // not have a directory component (example: foo.o becomes foo)
        llvm::SmallString<128> objectPath( st.filename.c_str() );
        invokeLinker( argv[0], dirWithStem, objectPath, (void*)&main, st );
        return (int)ReturnCodes::success;
    }

    serializeModuleMLIR( mod, flags, dirWithStem );

    if ( llvmDEBUG && mlir::failed( mlir::verify( mod ) ) )
    {
        llvm::errs() << COMPILER_NAME ": error: MLIR failed verification\n";
        mod->dump();
        std::exit( (int)ReturnCodes::verifyError );
    }

    lowerAssembleAndLinkModule( mod, dirWithStem, st, flags, argv[0], (void*)&main );

    return (int)ReturnCodes::success;
}

// vim: et ts=4 sw=4
