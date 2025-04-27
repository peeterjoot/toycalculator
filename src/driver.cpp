#include <assert.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>
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
#include "ToyParser.h"
#include "ToyPasses.h"
#include "ToyToLLVMLowering.h"

// Define a category for Toy Calculator options
static llvm::cl::OptionCategory ToyCategory( "Toy Calculator Options" );

// Command-line option for input file
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc( "<input file>" ),
    llvm::cl::init( "-" ), llvm::cl::value_desc( "filename" ),
    llvm::cl::cat( ToyCategory ), llvm::cl::NotHidden );

static llvm::cl::opt<bool> enableLocation(
    "location", llvm::cl::desc( "Enable MLIR location output" ),
    llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

static llvm::cl::opt<std::string> outDir(
    "output-directory", // Option name (used as --outdir)
    llvm::cl::desc("Output directory for generated files (e.g., .mlir, .ll, .o)"),
    llvm::cl::value_desc("directory"), // Placeholder in help text
    llvm::cl::init(""), // Default value (empty string, meaning the path to the source file)
    llvm::cl::cat(ToyCategory) // Group with other Toy options
);

// Add command-line option for LLVM IR emission
static llvm::cl::opt<bool> emitLLVM( "emit-llvm",
                                     llvm::cl::desc( "Emit LLVM IR" ),
                                     llvm::cl::init( false ),
                                     llvm::cl::cat( ToyCategory ) );

// Add command-line option for MLIR emission
static llvm::cl::opt<bool> emitMLIR( "emit-mlir",
                                     llvm::cl::desc( "Emit MLIR IR" ),
                                     llvm::cl::init( false ),
                                     llvm::cl::cat( ToyCategory ) );

// Noisy debugging output
static llvm::cl::opt<bool> llvmDEBUG(
    "debug-llvm",
    llvm::cl::desc( "Include MLIR dump, and turn off multithreading" ),
    llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

enum class return_codes : int
{
    success,
    cannot_open_file,
    semantic_error,
    unknown_error
};

using namespace toy;

int main( int argc, char **argv )
{
    // Initialize LLVM targets for code generation.
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Calculator compiler\n" );

    std::ifstream inputStream;
    std::string filename = inputFilename;
    if ( filename != "-" )
    {
        inputStream.open( filename );
        if ( !inputStream.is_open() )
        {
            llvm::errs() << std::format( "Error: Cannot open file {}\n",
                                         filename );
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

        antlr4::tree::ParseTree *tree = parser.startRule();
        antlr4::tree::ParseTreeWalker::DEFAULT.walk( &listener, tree );

        // For now, always dump the original MLIR unconditionally, even if we
        // are doing the LLVM IR lowering pass:
        mlir::OpPrintingFlags flags;
        if ( enableLocation )
        {
            flags.printGenericOpForm().enableDebugInfo( true );
        }
        llvm::StringRef stem = llvm::sys::path::stem( filename );
        if ( stem.empty() )
        {
            throw std::runtime_error( "Invalid filename: empty stem: '" +
                                      filename + "'" );
        }
        llvm::StringRef dirname = llvm::sys::path::parent_path( filename );
        llvm::SmallString<128> dirWithStem;
        if ( outDir != "" )
        {
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
            llvm::SmallString<128> path = dirWithStem;
            path += ".mlir";
            std::error_code EC;
            llvm::raw_fd_ostream out( path.str(), EC,
                                      llvm::sys::fs::OF_Text );
            if ( EC )
                throw std::runtime_error( "Failed to open file: " +
                                          EC.message() );

            listener.getModule().print( out, flags );
        }

        auto module = listener.getModule();
        auto context = module.getContext();

        // Register dialect translations
        mlir::registerLLVMDialectTranslation( *context );
        mlir::registerBuiltinDialectTranslation( *context );

        if ( llvmDEBUG )
            context->disableMultithreading( true );

        // Set up pass manager for lowering
        mlir::PassManager pm( context );
        if ( llvmDEBUG )
            pm.enableIRPrinting();

        pm.addPass( mlir::createToyToLLVMLoweringPass() );
        pm.addPass( mlir::createConvertSCFToCFPass() );
        pm.addPass( mlir::createConvertFuncToLLVMPass() );
        // pm.addPass(mlir::createConvertMemRefToLLVMPass());
        pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
        // pm.addPass(mlir::createConvertArithToLLVMPass());
        pm.addPass( mlir::createConvertControlFlowToLLVMPass() );

        if ( llvm::failed( pm.run( module ) ) )
            throw std::runtime_error( "LLVM lowering failed" );

        // Export to LLVM IR
        llvm::LLVMContext llvmContext;
        auto llvmModule =
            mlir::translateModuleToLLVMIR( module, llvmContext, filename );
        if ( !llvmModule )
            throw std::runtime_error( "Failed to translate to LLVM IR" );

        if ( emitLLVM )
        {
            llvm::SmallString<128> path = dirWithStem;
            path += ".ll";
            std::error_code EC;
            llvm::raw_fd_ostream out( path.str(), EC,
                                      llvm::sys::fs::OF_Text );
            if ( EC )
                throw std::runtime_error( "Failed to open file: " +
                                          EC.message() );

            llvmModule->print( out, nullptr );
        }
    }
    catch ( const semantic_exception &e )
    {
        // already printed the message.  return something non-zero
        return (int)return_codes::semantic_error;
    }
    catch ( const std::exception &e )
    {
        llvm::errs() << std::format( "FATAL ERROR: {}\n", e.what() );
        return (int)return_codes::unknown_error;
    }

    return (int)return_codes::success;
}

// vim: et ts=4 sw=4
