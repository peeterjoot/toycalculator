/// @file    driver.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This is the compiler driver for the silly compiler (originally "toy calculator").
///
/// Orchestration of compilation and linking, and command line argument handling.
///
/// - command line options handling,
/// - runs the antlr4 parse tree listener (w/ MLIR builder),
/// - runs the LLVM-IR lowering pass, and
/// - runs the assembly printer.
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>    // InitializeAllTargetInfos

#include "CompilationUnit.hpp"
#include "DialectContext.hpp"
#include "DriverState.hpp"
#include "OptLevel.hpp"
#include "ReturnCodes.hpp"
#include "SourceManager.hpp"

/// --debug- class for the driver
#define DEBUG_TYPE "silly-driver"

// Define a category for silly compiler options
static llvm::cl::OptionCategory SillyCategory( "Silly Compiler Options" );

////////////////////////////////////////////////////////////////////////////////////////
//
// Options related to output files
//

static llvm::cl::list<std::string> inputFilenames( llvm::cl::Positional, llvm::cl::desc( "<input file(s)>" ),
                                                   llvm::cl::OneOrMore, llvm::cl::value_desc( "filename" ),
                                                   llvm::cl::cat( SillyCategory ), llvm::cl::NotHidden );

static llvm::cl::opt<bool> compileOnly( "c", llvm::cl::desc( "Compile only and don't link." ), llvm::cl::init( false ),
                                        llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> assembleOnly( "S",
                                         llvm::cl::desc( "Assemble only; emit silly dialect textual MLIR and stop." ),
                                         llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> keepTemps( "keep-temp", llvm::cl::desc( "Do not automatically delete temporary files." ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<std::string> outDir(
    "output-directory", llvm::cl::desc( "Output directory for generated files (e.g., .mlir, .ll, .o)" ),
    llvm::cl::value_desc( "directory" ), llvm::cl::init( "" ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<std::string> oName( "o", llvm::cl::desc( "Executable or object name" ),
                                         llvm::cl::value_desc( "filename" ), llvm::cl::init( "" ),
                                         llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<std::string> imports( "imports",
                                           llvm::cl::desc( "comma separated list of pre-compiled import modules" ),
                                           llvm::cl::value_desc( "csv" ), llvm::cl::init( "" ),
                                           llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> emitMLIR(
    "emit-mlir",
    llvm::cl::desc( "Emit MLIR IR for the silly dialect.  Text .mlir format by default, and .mlirbc with -c" ),
    llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> emitMLIRBC( "emit-mlirbc", llvm::cl::desc( "Emit MLIR BC for the silly dialect" ),
                                       llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

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

static llvm::cl::opt<silly::OptLevel> optLevel(
    "O", llvm::cl::desc( "Optimization level" ),
    llvm::cl::values( clEnumValN( silly::OptLevel::O0, "0", "No optimization" ),
                      clEnumValN( silly::OptLevel::O1, "1", "Light optimization" ),
                      clEnumValN( silly::OptLevel::O2, "2", "Moderate optimization" ),
                      clEnumValN( silly::OptLevel::O3, "3", "Aggressive optimization" ) ),
    llvm::cl::init( silly::OptLevel::O0 ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> noAbortPath( "no-abort-path",
                                        llvm::cl::desc( "Specify to omit include source path in ABORT" ),
                                        llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> debugInfo( "g",
                                      llvm::cl::desc( "Enable location output in MLIR dumps, and dwarf metadata "
                                                      "creation in the lowered LLVM IR)" ),
                                      llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

////////////////////////////////////////////////////////////////////////////////////////
//
// Diagnostic options:
//
static llvm::cl::opt<bool> verboseLink( "verbose-link", llvm::cl::desc( "Display the link command line on stderr" ),
                                        llvm::cl::init( false ), llvm::cl::cat( SillyCategory ) );

static llvm::cl::opt<bool> noVerboseParseError( "no-verbose-parse-error",
                                                llvm::cl::desc( "Hide grammar specific context for parse errors" ),
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

namespace silly
{
    /// Assuming that a message has already been displayed, return with a non-zero return code.
    void fatalDriverError( ReturnCodes rc )
    {
        assert( (int)rc < 256 );    // Assume unix.

        std::exit( (int)rc );
    }
}    // namespace silly

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

    ds.compileOnly = compileOnly;
    ds.assembleOnly = assembleOnly;
    ds.keepTemps = keepTemps;
    ds.emitMLIR = emitMLIR;
    ds.emitMLIRBC = emitMLIRBC;
    if ( ds.assembleOnly )
    {
        ds.emitMLIR = true;
    }
    if ( ds.emitMLIR and ds.compileOnly )
    {
        ds.emitMLIR = false;
        ds.emitMLIRBC = true;
    }
    ds.emitLLVM = emitLLVM;
    ds.toStdout = toStdout;
    ds.noAbortPath = noAbortPath;
    ds.debugInfo = debugInfo;
    ds.verboseLink = verboseLink;
    ds.noVerboseParseError = noVerboseParseError;
    ds.llvmDEBUG = llvmDEBUG;
    ds.noColorErrors = noColorErrors;

    ds.outDir = outDir;
    ds.oName = oName;
    ds.imports = imports;
    ds.initFillValue = (uint8_t)initFillValue;
    switch ( optLevel )
    {
        case silly::OptLevel::O0:
            ds.opt = llvm::OptimizationLevel::O0;
            break;
        case silly::OptLevel::O1:
            ds.opt = llvm::OptimizationLevel::O1;
            break;
        case silly::OptLevel::O2:
            ds.opt = llvm::OptimizationLevel::O2;
            break;
        case silly::OptLevel::O3:
            ds.opt = llvm::OptimizationLevel::O3;
            break;
    }

    if ( ds.compileOnly and ds.assembleOnly )
    {
        llvm::errs() << COMPILER_NAME ": error: command line options -S and -c cannot be used together\n";
        silly::fatalDriverError( ReturnCodes::badOption );
    }

    std::vector<std::string>& files = inputFilenames;

    silly::SourceManager sm( ds, &dialectLoader.context, files[0] );

    if ( !ds.imports.empty() )
    {
        auto& cup = sm.createCU( ds.imports );

        // Go as far as mlir::ModuleOp creation, but don't lower to llvm (yet):
        sm.createAndSerializeMLIR( cup );
    }

    for ( const auto& filename : files )
    {
        auto& cup = sm.createCU( filename );

        sm.createAndSerializeMLIR( cup );

        bool moreToDo = sm.createAndSerializeLLVM( cup );

        if ( moreToDo )
        {
            sm.serializeObject( cup );
        }
    }

    if ( !ds.imports.empty() )
    {
        auto & cup = sm.findCU( ds.imports );

        bool moreToDo = sm.createAndSerializeLLVM( cup );

        if ( moreToDo )
        {
            sm.serializeObject( cup );
        }
    }

    sm.link();

    return (int)ReturnCodes::success;
}

// vim: et ts=4 sw=4
