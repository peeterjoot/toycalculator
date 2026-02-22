/// @file    CompilationUnit.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler handling of a single compilation unit.
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

#include "CompilationUnit.hpp"
#include "DialectContext.hpp"
#include "DriverState.hpp"
#include "OptLevel.hpp"
#include "ParseListener.hpp"
#include "ReturnCodes.hpp"
#include "SillyDialect.hpp"
#include "SillyPasses.hpp"
#include "createSillyToLLVMLoweringPass.hpp"

// TODO:
// Reduce use of raw ModuleOp -- prefer passing OwningOpRef& or keep it local

#define DEBUG_TYPE "silly-cu"

namespace silly
{
    void fatalDriverError( ReturnCodes rc );

    CompilationUnit::CompilationUnit( silly::DriverState& d, std::string f, mlir::MLIRContext* c )
        : ds{ d }, filename{ f }, context{ c }
    {
        if ( ds.debugInfo )
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
            if ( mlir::failed( mlir::verify( mod ) ) )
            {
                llvm::errs() << COMPILER_NAME ": error: MLIR failed verification\n";
                mod->dump();
                fatalDriverError( ReturnCodes::verifyError );
            }

            serializeModuleMLIR();
        }
    }

    void CompilationUnit::mlirToLLVM()
    {
        mlir::ModuleOp mod = rmod.get();

        // Register dialect translations
        mlir::registerLLVMDialectTranslation( *context );
        mlir::registerBuiltinDialectTranslation( *context );

        if ( ds.llvmDEBUG )
        {
            context->disableMultithreading( true );
        }

        // Set up pass manager for lowering
        mlir::PassManager pm( context );
        if ( ds.llvmDEBUG )
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
        if ( ds.llvmDEBUG )
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

        if ( ds.toStdout )
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

        llvm::ModulePassManager MPM = passBuilder.buildPerModuleDefaultPipeline( ds.opt );

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

    void CompilationUnit::constructObjectPath( llvm::SmallString<128>& outputFilename )
    {
        outputFilename += dirWithStem;
        outputFilename += ".o";
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

    void CompilationUnit::serializeModuleMLIR()
    {
        if ( ds.emitMLIR )
        {
            if ( ds.toStdout )
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
                    llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file {}: {}\n",
                                                 std::string( path ), EC.message() );
                    fatalDriverError( ReturnCodes::openError );
                }
                rmod->print( out, flags );
            }
        }
    }

    void CompilationUnit::serializeModuleLLVMIR()
    {
        // Dump the pre-optimized LL if we aren't creating a .o
        if ( !ds.emitLLVM )
        {
            return;
        }

        if ( ds.toStdout )
        {
            llvmModule->print( llvm::outs(), nullptr, ds.debugInfo /* print debug info */ );
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
                llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open file '{}': {}\n",
                                             std::string( path ), EC.message() );
                fatalDriverError( ReturnCodes::openError );
            }

            llvmModule->print( out, nullptr, ds.debugInfo /* print debug info */ );
        }
    }

    void CompilationUnit::makeOutputDirectory()
    {
        llvm::StringRef dirname = llvm::sys::path::parent_path( filename );
        // Create output directory if specified
        if ( !ds.outDir.empty() )
        {
            std::error_code EC = llvm::sys::fs::create_directories( ds.outDir );
            if ( EC )
            {
                llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create output directory '{}': {}\n",
                                             ds.outDir, EC.message() );
                fatalDriverError( ReturnCodes::directoryError );
            }

            outdir = ds.outDir;
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
}    // namespace silly
