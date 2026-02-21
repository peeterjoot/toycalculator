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
#include <llvm/Support/Program.h>
#include <llvm/Support/TargetSelect.h>

#include <format>
#include <fstream>

#include "CompilationUnit.hpp"
#include "DialectContext.hpp"
#include "DriverState.hpp"
#include "OptLevel.hpp"
#include "ParseListener.hpp"
#include "ReturnCodes.hpp"

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
        silly::fatalDriverError( ReturnCodes::filenameParseError );
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
        silly::fatalDriverError( ReturnCodes::linkError );
    }
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
    ds.isOptimized = optLevel != silly::OptLevel::O0 ? true : false;
    ds.fillValue = (uint8_t)initFillValue;
    ds.debugInfo = debugInfo;
    ds.colorErrors = !noColorErrors;
    ds.abortOmitPath = noAbortPath;
    ds.toStdout = toStdout;
    ds.emitMLIR = emitMLIR;
    ds.emitLLVM = emitLLVM;
    ds.llvmDEBUG = llvmDEBUG;
    ds.outDir = outDir;
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

    std::vector<std::string> objFiles;
    std::vector<std::string> tmpToDelete;

    llvm::SmallString<128> exeName;

    for ( const auto& filename : inputFilenames )
    {
        silly::CompilationUnit st( ds, filename, &dialectLoader.context );

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

        if ( st.getInputType() == silly::InputType::OBJECT )
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

                    silly::fatalDriverError( ReturnCodes::tempCreationError );
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
