#include <assert.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
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
    "location", llvm::cl::desc( "Enable location output (MLIR and LLVM IR)" ),
    llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

static llvm::cl::opt<std::string> outDir(
    "output-directory",
    llvm::cl::desc(
        "Output directory for generated files (e.g., .mlir, .ll, .o)" ),
    llvm::cl::value_desc( "directory" ), llvm::cl::init( "" ),
    llvm::cl::cat( ToyCategory ) );

// Add command-line option for MLIR emission
static llvm::cl::opt<bool> emitMLIR( "emit-mlir",
                                     llvm::cl::desc( "Emit MLIR IR" ),
                                     llvm::cl::init( false ),
                                     llvm::cl::cat( ToyCategory ) );

// Add command-line option for LLVM IR emission
static llvm::cl::opt<bool> emitLLVM( "emit-llvm",
                                     llvm::cl::desc( "Emit LLVM IR" ),
                                     llvm::cl::init( false ),
                                     llvm::cl::cat( ToyCategory ) );

// Add command-line option for object file emission
static llvm::cl::opt<bool> emitObject(
    "emit-object", llvm::cl::desc( "Emit object file (.o)" ),
    llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

// Noisy debugging output
static llvm::cl::opt<bool> llvmDEBUG(
    "debug-llvm",
    llvm::cl::desc( "Include MLIR dump, and turn off multithreading" ),
    llvm::cl::init( false ), llvm::cl::cat( ToyCategory ) );

enum class OptLevel : int
{
    O0,
    O1,
    O2,
    O3
};

static llvm::cl::opt<OptLevel> optLevel(
    "O", llvm::cl::desc( "Optimization level" ),
    llvm::cl::values( clEnumValN( OptLevel::O0, "0", "No optimization" ),
                      clEnumValN( OptLevel::O1, "1", "Light optimization" ),
                      clEnumValN( OptLevel::O2, "2", "Moderate optimization" ),
                      clEnumValN( OptLevel::O3, "3",
                                  "Aggressive optimization" ) ),
    llvm::cl::init( OptLevel::O0 ), llvm::cl::cat( ToyCategory ) );

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

        // Create output directory if specified
        if ( !outDir.empty() )
        {
            std::error_code EC = llvm::sys::fs::create_directories( outDir );
            if ( EC )
            {
                throw std::runtime_error(
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
            llvm::SmallString<128> path = dirWithStem;
            path += ".mlir";
            std::error_code EC;
            llvm::raw_fd_ostream out( path.str(), EC, llvm::sys::fs::OF_Text );
            if ( EC )
            {
                throw std::runtime_error( "Failed to open file: " +
                                          EC.message() );
            }
            listener.getModule().print( out, flags );
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

        pm.addPass( mlir::createToyToLLVMLoweringPass() );
        pm.addPass( mlir::createConvertSCFToCFPass() );
        pm.addPass( mlir::createConvertFuncToLLVMPass() );
        pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
        pm.addPass( mlir::createConvertControlFlowToLLVMPass() );

        if ( llvm::failed( pm.run( module ) ) )
        {
            throw std::runtime_error( "LLVM lowering failed" );
        }

        // Export to LLVM IR
        llvm::LLVMContext llvmContext;
        auto llvmModule =
            mlir::translateModuleToLLVMIR( module, llvmContext, filename );
        if ( !llvmModule )
        {
            throw std::runtime_error( "Failed to translate to LLVM IR" );
        }

        if ( emitLLVM || emitObject )
        {
            if ( emitLLVM )
            {
                llvm::SmallString<128> path = dirWithStem;
                path += ".ll";
                std::error_code EC;
                llvm::raw_fd_ostream out( path.str(), EC,
                                          llvm::sys::fs::OF_Text );
                if ( EC )
                {
                    throw std::runtime_error( "Failed to open file: " +
                                              EC.message() );
                }
                if ( enableLocation )
                {
                    // Verify the module to ensure debug info is valid
                    if ( llvm::verifyModule( *llvmModule, &llvm::errs() ) )
                    {
                        throw std::runtime_error( "Invalid LLVM IR module" );
                    }
                    // Print with debug info (metadata is included by default)
                    llvmModule->print( out, nullptr,
                                       true /* print debug info */ );
                }
                else
                {
                    llvmModule->print( out, nullptr );
                }
            }

            if ( emitObject )
            {
                // Set target triple
                std::string targetTriple = llvm::sys::getProcessTriple();
                llvmModule->setTargetTriple( targetTriple );

                // Lookup the target
                std::string error;
                const llvm::Target *target =
                    llvm::TargetRegistry::lookupTarget( targetTriple, error );
                if ( !target )
                {
                    throw std::runtime_error( "Failed to find target: " +
                                              error );
                }

                // Create the target machine
                std::unique_ptr<llvm::TargetMachine> targetMachine(
                    target->createTargetMachine( targetTriple, "generic", "",
                                                 llvm::TargetOptions(),
                                                 std::nullopt ) );
                if ( !targetMachine )
                {
                    throw std::runtime_error(
                        "Failed to create target machine" );
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
                llvm::ModulePassManager MPM =
                    passBuilder.buildPerModuleDefaultPipeline( opt );

                MPM.run( *llvmModule, MAM );

                // Emit object file
                llvm::SmallString<128> outputFilename( dirWithStem );
                outputFilename += ".o";
                std::error_code EC;
                llvm::raw_fd_ostream dest( outputFilename.str(), EC,
                                           llvm::sys::fs::OF_None );
                if ( EC )
                {
                    throw std::runtime_error( "Failed to open output file: " +
                                              EC.message() );
                }

                llvmModule->setDataLayout( targetMachine->createDataLayout() );
                llvm::legacy::PassManager codegenPM;
                if ( targetMachine->addPassesToEmitFile(
                         codegenPM, dest, nullptr,
                         llvm::CodeGenFileType::ObjectFile ) )
                {
                    throw std::runtime_error(
                        "TargetMachine can't emit an object file" );
                }

                codegenPM.run( *llvmModule );
                dest.close();

                llvm::outs()
                    << "Generated object file: " << outputFilename << "\n";
            }
        }
    }
    catch ( const semantic_exception &e )
    {
        // already printed the message. return something non-zero
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
