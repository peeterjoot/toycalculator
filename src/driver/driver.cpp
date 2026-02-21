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

#include "DialectContext.hpp"
#include "DriverState.hpp"
#include "ParseListener.hpp"
#include "SillyDialect.hpp"
#include "SillyPasses.hpp"
#include "createSillyToLLVMLoweringPass.hpp"

// TODO:
// Reduce use of raw ModuleOp — prefer passing OwningOpRef& or keep it local

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
    tempCreationError,
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

/// Track everything related to a specific compilation unit through all phases of the compilation pipeline.
///
/// This class manages the state and orchestrates the transformation of a single source file
/// through parsing, MLIR generation, LLVM IR lowering, optimization, and object code emission.
class CompilationUnit
{
   public:
    /// Construct a new module state for a compilation unit.
    ///
    /// @param f The source filename to compile
    /// @param c The MLIR context for this compilation
    CompilationUnit( silly::DriverState& d, std::string f, mlir::MLIRContext* c );

    /// Parse the source file and build the MLIR module.
    ///
    /// Determines input type (.silly, .mlir, .o) and invokes the appropriate parser.
    /// Populates rmod with the resulting MLIR module.
    void processSourceFile();

    /// Lower the MLIR module to LLVM IR dialect.
    ///
    /// Runs the MLIR-to-LLVM lowering passes and translates the result to an llvm::Module.
    /// Populates llvmModule as a side effect.
    void mlirToLLVM();

    /// Run LLVM optimization passes on the lowered module.
    ///
    /// Creates a target machine and runs the optimization pipeline based on the -O level.
    /// Modifies llvmModule in place.
    void runOptimizationPasses();

    /// Emit object code (.o file) from the optimized LLVM module.
    ///
    /// @param outputFilename[out] Path where the object file will be written
    void serializeObjectCode( const llvm::SmallString<128>& outputFilename );

    /// Get the detected input file type.
    ///
    /// @return The input type (.silly, .mlir, .o, or unknown)
    InputType getInputType() const
    {
        return ity;
    }

    /// Construct the output path for the object file.
    ///
    /// @param outputFilename[out] Buffer to receive the constructed path
    void constructObjectPath( llvm::SmallString<128>& outputFilename );

    /// Return the executable path associated with this file.
    const llvm::SmallString<128>& getDefaultExecutablePath() const
    {
        return dirWithStem;
    }

    const llvm::SmallString<128>& getOutputDirectory() const
    {
        return outdir;
    }

   private:
    silly::DriverState& ds;

    /// Source filename being compiled
    std::string filename{};

    /// MLIR context for this compilation unit
    mlir::MLIRContext* context{};

    /// The MLIR module (either parsed or generated from source)
    mlir::OwningOpRef<mlir::ModuleOp> rmod{};

    /// Detected input file type (.silly, .mlir, or .o)
    InputType ity{};

    /// Flags controlling MLIR printing (for debug output)
    mlir::OpPrintingFlags flags;

    /// LLVM context - must persist for lifetime of llvmModule
    llvm::LLVMContext llvmContext;

    /// The LLVM IR module after lowering from MLIR
    std::unique_ptr<llvm::Module> llvmModule;

    /// Target machine for code generation and optimization
    std::unique_ptr<llvm::TargetMachine> targetMachine;

    /// Output directory combined with filename stem (no extension)
    llvm::SmallString<128> dirWithStem;

    /// Just the output directory
    llvm::SmallString<128> outdir;

    /// Determine the input type from a filename extension.
    ///
    /// @param filename The filename to examine
    /// @return The detected input type
    static InputType getInputType( llvm::StringRef filename );

    /// Serialize the MLIR module to a .mlir file (if --emit-mlir specified).
    void serializeModuleMLIR();

    /// Serialize the LLVM IR module to a .ll file (if --emit-llvm specified).
    void serializeModuleLLVMIR();

    /// Create the directory named in --output-directory.
    ///
    /// Populates outdir as a side effect with the output directory, or the directory
    /// part of the filename path (if specified), or an empty string.
    void makeOutputDirectory();

    /// Parse a .mlir file into the MLIR module.
    ///
    /// Saves the parsed module to rmod as a side effect.
    void parseMLIRFile();
};

////////////////////////////////////////////////////////////////////////////////////////
//
// Options related to output files
//

static llvm::cl::list<std::string> inputFilenames( llvm::cl::Positional, llvm::cl::desc( "<input file(s)>" ),
                                                   llvm::cl::OneOrMore, llvm::cl::value_desc( "filename" ),
                                                   llvm::cl::cat( SillyCategory ), llvm::cl::NotHidden );

static llvm::cl::opt<bool> compileOnly( "c", llvm::cl::desc( "Compile only and don't link." ), llvm::cl::init( false ),
                                        llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> keepTemps( "keep-temp", llvm::cl::desc( "Do not automatically delete temporary files." ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<std::string> outDir(
    "output-directory", llvm::cl::desc( "Output directory for generated files (e.g., .mlir, .ll, .o)" ),
    llvm::cl::value_desc( "directory" ), llvm::cl::init( "" ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<std::string> oName( "o", llvm::cl::desc( "Executable or object name" ),
                                         llvm::cl::value_desc( "filename" ), llvm::cl::init( "" ),
                                         llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> emitMLIR( "emit-mlir", llvm::cl::desc( "Emit MLIR IR" ), llvm::cl::init( false ),
                                     llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> emitLLVM( "emit-llvm", llvm::cl::desc( "Emit LLVM IR" ), llvm::cl::init( false ),
                                     llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> toStdout( "stdout", llvm::cl::desc( "LLVM and MLIR on stdout instead of to a file" ),
                                     llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

////////////////////////////////////////////////////////////////////////////////////////
//
// Options that change code generation:
//
static llvm::cl::opt<int> initFillValue( "init-fill", llvm::cl::desc( "Initializer fill value." ), llvm::cl::init( 0 ),
                                         llvm::cl::ValueRequired, llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<OptLevel> optLevel( "O", llvm::cl::desc( "Optimization level" ),
                                         llvm::cl::values( clEnumValN( OptLevel::O0, "0", "No optimization" ),
                                                           clEnumValN( OptLevel::O1, "1", "Light optimization" ),
                                                           clEnumValN( OptLevel::O2, "2", "Moderate optimization" ),
                                                           clEnumValN( OptLevel::O3, "3", "Aggressive optimization" ) ),
                                         llvm::cl::init( OptLevel::O0 ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> noAbortPath( "no-abort-path",
                                        llvm::cl::desc( "Specify to omit include source path in ABORT" ),
                                        llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> debugInfo( "g",
                                      llvm::cl::desc( "Enable location output in MLIR, and dwarf metadata "
                                                      "creation in the lowered LLVM IR)" ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

////////////////////////////////////////////////////////////////////////////////////////
//
// Diagnostic options:
//
static llvm::cl::opt<bool> verboseLink( "verbose-link", llvm::cl::desc( "Display the link command line on stderr" ),
                                        llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

// Noisy debugging output (this is different than --debug which is intercepted by llvm itself)
static llvm::cl::opt<bool> llvmDEBUG( "debug-llvm", llvm::cl::desc( "Include MLIR dump, and turn off multithreading" ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> noColorErrors( "no-color-errors", llvm::cl::desc( "Disable color error messages" ),
                                          llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

////////////////////////////////////////////////////////////////////////////////////////
//
// Helper functions
//

/// Assuming that a message has already been displayed, return with a non-zero return code.
static void fatalDriverError( ReturnCodes rc )
{
    assert( (int)rc < 256 );    // Assume unix.

    std::exit( (int)rc );
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

static void showLinkCommand( const std::string& linker, llvm::SmallVector<llvm::StringRef, 24>& args )
{
    llvm::errs() << "# " << linker;

    for ( const auto& a : args )
    {
        llvm::errs() << a << ' ';
    }

    llvm::errs() << '\n';
}

/// Invoke the system linker to create an executable.
///
/// @param objectPath Path to the object file to link
static void invokeLinker( const llvm::SmallString<128>& exePath, const std::vector<std::string>& objectPaths,
                          silly::DriverState& ds )
{
    // Get the driver path
    std::string driver = llvm::sys::fs::getMainExecutable( ds.argv0, ds.mainSymbol );
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
        fatalDriverError( ReturnCodes::filenameParseError );
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
    std::vector<std::string> linkerArgValues;
    linkerArgValues.push_back( std::string( linkerPath.get() ) );
    linkerArgValues.push_back( "-g" );
    linkerArgValues.push_back( "-o" );
    linkerArgValues.push_back( std::string( exePath ) );
    for ( const auto& o : objectPaths )
    {
        linkerArgValues.push_back( o );
    }
    linkerArgValues.push_back( "-L" );
    linkerArgValues.push_back( std::string( libPath ) );
    linkerArgValues.push_back( "-l" );
    linkerArgValues.push_back( "silly_runtime" );
    linkerArgValues.push_back( std::string( rpathOption ) );
    if ( ds.needsMathLib )
    {
        linkerArgValues.push_back( "-lm" );
    }

    llvm::SmallVector<llvm::StringRef, 24> linkerArgs;
    for ( const auto& s : linkerArgValues )
    {
        linkerArgs.push_back( s );
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
        fatalDriverError( ReturnCodes::linkError );
    }
}

////////////////////////////////////////////////////////////////////////////////////////
//
// CompilationUnit members (FIXME) reorder these more sensibly.
//
CompilationUnit::CompilationUnit( silly::DriverState& d, std::string f, mlir::MLIRContext* c )
    : ds{ d }, filename{ f }, context{ c }
{
    if ( debugInfo )
    {
        flags.enableDebugInfo( true );
    }

    makeOutputDirectory();

    llvm::StringRef stem = llvm::sys::path::stem( filename );    // foo/bar.silly -> stem: is just bar, not foo/bar
    if ( stem.empty() )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Invalid filename '{}', empty stem\n", filename );
        fatalDriverError( ReturnCodes::filenameParseError );
    }

    dirWithStem = outdir;
    if ( dirWithStem.empty() )
    {
        dirWithStem = stem;
    }
    else
    {
        llvm::sys::path::append( dirWithStem, stem );
    }
}

void CompilationUnit::serializeModuleLLVMIR()
{
    // Dump the pre-optimized LL if we aren't creating a .o
    if ( !emitLLVM )
    {
        return;
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
            // FIXME: probably want llvm::formatv here and elsewhere to avoid the std::string casting hack (assuming
            // it knows how to deal with StringRef)
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open file '{}': {}\n", std::string( path ),
                                         EC.message() );
            fatalDriverError( ReturnCodes::openError );
        }

        llvmModule->print( out, nullptr, debugInfo /* print debug info */ );
    }
}

void CompilationUnit::runOptimizationPasses()
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
        fatalDriverError( ReturnCodes::loweringError );
    }

    // Create the target machine
    targetMachine.reset(
        target->createTargetMachine( targetTriple, "generic", "", llvm::TargetOptions(), std::nullopt ) );

    if ( !targetMachine )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create target machine\n" );
        fatalDriverError( ReturnCodes::loweringError );
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
}

void CompilationUnit::serializeObjectCode( const llvm::SmallString<128>& outputFilename )
{
    std::error_code EC;
    llvm::raw_fd_ostream dest( outputFilename.str(), EC, llvm::sys::fs::OF_None );
    if ( EC )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open output file '{}': {}\n",
                                     std::string( outputFilename ), EC.message() );
        fatalDriverError( ReturnCodes::openError );
    }

    llvmModule->setDataLayout( targetMachine->createDataLayout() );
    llvm::legacy::PassManager codegenPM;
    if ( targetMachine->addPassesToEmitFile( codegenPM, dest, nullptr, llvm::CodeGenFileType::ObjectFile ) )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: TargetMachine can't emit an object file\n" );
        fatalDriverError( ReturnCodes::loweringError );
    }

    codegenPM.run( *llvmModule );
    dest.close();

    LLVM_DEBUG( { llvm::outs() << "Generated object file: " << outputFilename << '\n'; } );
}

void CompilationUnit::mlirToLLVM()
{
    mlir::ModuleOp mod = rmod.get();

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

    pm.addPass( mlir::createSillyToLLVMLoweringPass( &ds ) );
    pm.addPass( mlir::createSCFToControlFlowPass() );
    pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
    pm.addPass( mlir::createConvertControlFlowToLLVMPass() );

    if ( llvm::failed( pm.run( mod ) ) )
    {
        llvm::errs() << "IR after stage I lowering failure:\n";
        mod->dump();
        llvm::errs() << COMPILER_NAME ": error: Stage I LLVM lowering failed\n";
        fatalDriverError( ReturnCodes::loweringError );
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
        fatalDriverError( ReturnCodes::loweringError );
    }

    if ( toStdout )
    {
        llvm::outs() << "Before module lowering:\n";
        mod.print( llvm::outs(), flags );
    }

    // The module should now contain mostly LLVM-IR instructions, with the exception of the top level module,
    // and the MLIR style loc() references.  Those last two MLIR artifacts will be convered to LLVM-IR
    // now, also producing !DILocation's for all the loc()s.
    llvmModule = mlir::translateModuleToLLVMIR( mod, llvmContext, filename );

    if ( !llvmModule )
    {
        llvm::errs() << COMPILER_NAME ": error: Failed to translate to LLVM IR\n";
        fatalDriverError( ReturnCodes::loweringError );
    }

    // Verify the module to ensure debug info is valid
    if ( llvm::verifyModule( *llvmModule, &llvm::errs() ) )
    {
        llvm::errs() << COMPILER_NAME ": error: Invalid LLVM IR module\n";
        fatalDriverError( ReturnCodes::loweringError );
    }

    serializeModuleLLVMIR();
}

void CompilationUnit::makeOutputDirectory()
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
            fatalDriverError( ReturnCodes::directoryError );
        }

        outdir = outDir;
    }
    else if ( dirname != "" )
    {
        outdir = dirname;
    }
}

void CompilationUnit::parseMLIRFile()
{
    auto fileOrErr = llvm::MemoryBuffer::getFile( filename );
    if ( std::error_code EC = fileOrErr.getError() )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file '{}': {}\n", filename, EC.message() );
        fatalDriverError( ReturnCodes::openError );
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer( std::move( *fileOrErr ), llvm::SMLoc{} );

    rmod = mlir::parseSourceFile<mlir::ModuleOp>( sourceMgr, context );
    if ( !rmod.get() )
    {
        llvm::errs() << std::format( COMPILER_NAME ": error: Failed to parse MLIR file '{}'\n", filename );
        fatalDriverError( ReturnCodes::parseError );
    }
}

void CompilationUnit::serializeModuleMLIR()
{
    if ( emitMLIR )
    {
        if ( toStdout )
        {
            rmod->print( llvm::outs(), flags );
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
                fatalDriverError( ReturnCodes::openError );
            }
            rmod->print( out, flags );
        }
    }
}

void CompilationUnit::processSourceFile()
{
    ity = getInputType( filename );
    if ( ity == InputType::Unknown )
    {
        llvm::errs() << std::format(
            COMPILER_NAME ": error: filename {} extension is none of .silly, .mlir/.sir, or .o\n", filename );
        fatalDriverError( ReturnCodes::badExtensionError );
    }

    mlir::ModuleOp mod{};

    if ( ity == InputType::Silly )
    {
        silly::ParseListener listener( ds, filename, context );

        rmod = listener.run();
        if ( ds.openFailed )
        {
            llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file {}\n", filename );
            fatalDriverError( ReturnCodes::openError );
        }

        mod = rmod.get();
        if ( !mod )
        {
            // should have already emitted diagnostics.
            fatalDriverError( ReturnCodes::parseError );
        }
    }
    else if ( ity == InputType::MLIR )
    {
        parseMLIRFile();
        mod = rmod.get();
    }

    if ( mod )
    {
        if ( llvmDEBUG && mlir::failed( mlir::verify( mod ) ) )
        {
            llvm::errs() << COMPILER_NAME ": error: MLIR failed verification\n";
            mod->dump();
            fatalDriverError( ReturnCodes::verifyError );
        }

        serializeModuleMLIR();
    }
}

InputType CompilationUnit::getInputType( llvm::StringRef filename )
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

void CompilationUnit::constructObjectPath( llvm::SmallString<128>& outputFilename )
{
    outputFilename += dirWithStem;
    outputFilename += ".o";
}

////////////////////////////////////////////////////////////////////////////////////////
//
// The driver entry point:
//
int main( int argc, char** argv )
{
    llvmInitialization( argc, argv );

    // once this goes out of scope, the module is toast and can't be referenced further.
    silly::DialectContext dialectLoader;

    silly::DriverState ds( argv[0], (void*)&main );
    ds.isOptimized = optLevel != OptLevel::O0 ? true : false;
    ds.fillValue = (uint8_t)initFillValue;
    ds.wantDebug = debugInfo;
    ds.colorErrors = !noColorErrors;
    ds.abortOmitPath = noAbortPath;
    std::vector<std::string> objFiles;
    std::vector<std::string> tmpToDelete;

    llvm::SmallString<128> exeName;

    for ( const auto& filename : inputFilenames )
    {
        CompilationUnit st( ds, filename, &dialectLoader.context );

        st.processSourceFile();

        if ( exeName.empty() )    // first source.
        {
            if ( oName.empty() )
            {
                // This exe-path should be split out from CompilationUnit, as it may not match the input source file
                // stem. The dirWithStem stuff is convoluted and confusing.
                exeName = st.getDefaultExecutablePath();
            }
            else
            {
                exeName = oName;
            }
        }

        llvm::SmallString<128> objectFilename;
        bool createdTemporary{};

        if ( st.getInputType() == InputType::OBJECT )
        {
            objectFilename = filename;
        }
        else
        {
            st.mlirToLLVM();

            st.runOptimizationPasses();

            if ( !oName.empty() && compileOnly )
            {
                objectFilename = oName;
            }
            else if ( compileOnly )
            {
                st.constructObjectPath( objectFilename );
            }
            else
            {
                const llvm::SmallString<128>& outdir = st.getOutputDirectory();
                llvm::SmallString<128> p;

                if ( outdir.empty() )
                {
                    llvm::SmallString<128> td;
                    llvm::sys::path::system_temp_directory( true, td );
                    p = td;
                }
                else
                {
                    p = outdir;
                }

                llvm::SmallString<128> o = llvm::sys::path::stem( filename );
                o += "-%%%%%%.o";
                llvm::sys::path::append( p, o );

                std::error_code EC;
                EC = llvm::sys::fs::createUniqueFile( p, objectFilename );
                if ( EC )
                {
                    // FIXME: another place to use formatv
                    llvm::errs() << std::format( COMPILER_NAME
                                                 ": error: Failed to create temporary object file in path '{}': {}\n",
                                                 std::string( p ), EC.message() );

                    fatalDriverError( ReturnCodes::tempCreationError );
                }

                if ( keepTemps )
                {
                    // FIXME: another place to use formatv
                    llvm::errs() << std::format( COMPILER_NAME ": info: created temporary: {}\n",
                                                 std::string( objectFilename ) );
                }

                createdTemporary = true;
            }


            st.serializeObjectCode( objectFilename );

            objFiles.push_back( std::string( objectFilename ) );

            if ( createdTemporary )
            {
                tmpToDelete.push_back( std::string( objectFilename ) );
            }
        }
    }

    if ( compileOnly == false )
    {
        invokeLinker( exeName, objFiles, ds );
    }

    if ( !keepTemps )
    {
        for ( const auto& filename : tmpToDelete )
        {
            llvm::sys::fs::remove( filename );
        }
    }

    return (int)ReturnCodes::success;
}

// vim: et ts=4 sw=4
