/// @file    CompilationUnit.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler handling of a single compilation unit.
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DebugProgramInstruction.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
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
#include <mlir/Bytecode/BytecodeWriter.h>
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
#include "SourceManager.hpp"
#include "createSillyToLLVMLoweringPass.hpp"

#define DEBUG_TYPE "silly-cu"

namespace silly
{
    CompilationUnit::CompilationUnit( silly::SourceManager& s )
        : sm{ s }, ds{ sm.getDriverState() }, context{ sm.getContext() }
    {
        if ( ds.debugInfo )
        {
            flags.enableDebugInfo( true );
        }
    }

    ReturnCodes CompilationUnit::processSourceFile( const std::string& sourceFileName )
    {
        ity = getInputType( sourceFileName );
        if ( ity == InputType::Unknown )
        {
            // coverage: bad-suffix-should-fail.silly
            llvm::errs() << std::format(
                COMPILER_NAME ": error: filename {} extension is none of .silly, .mlir/.sir, or .o\n", sourceFileName );
            return ReturnCodes::badExtensionError;
        }

        if ( ity == InputType::Silly )
        {
            silly::ParseListener listener( sm, sourceFileName );

            rmod = listener.run();
            if ( ds.openFailed )
            {
                // coverage: bad-file-should-fail.silly
                llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file {}\n", sourceFileName );
                return ReturnCodes::openError;
            }

            if ( !rmod )
            {
                // should have already emitted diagnostics.
                return ReturnCodes::parseError;
            }
        }
        else if ( ( ity == InputType::MLIR ) || ( ity == InputType::MLIRBC ) )
        {
            parseMLIRFile( sourceFileName );
        }
        else if ( ( ity == InputType::LLVMLL ) || ( ity == InputType::LLVMBC ) )
        {
            parseLLVMFile( sourceFileName );
        }

        if ( rmod )
        {
            if ( mlir::failed( mlir::verify( *rmod ) ) )
            {
                llvm::errs() << COMPILER_NAME ": error: MLIR failed verification\n";
                (*rmod).dump();
                return ReturnCodes::verifyError;
            }
        }

        return ReturnCodes::success;
    }

    ReturnCodes CompilationUnit::mlirToLLVM( const std::string& llvmSourceFilename )
    {
        if ( mlir::ModuleOp mod = rmod.get() )
        {
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
                return ReturnCodes::loweringError;
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
                return ReturnCodes::loweringError;
            }

            // The module should now contain mostly LLVM-IR instructions, with the exception of the top level module,
            // and the MLIR style loc() references.  Those last two MLIR artifacts will be convered to LLVM-IR
            // now, also producing !DILocation's for all the loc()s.
            //
            // The filename parameter is fed to these two lines, overriding the following default:
            //
            // ; ModuleID = 'LLVMDialectModule'
            // source_filename = "LLVMDialectModule"
            llvmModule = mlir::translateModuleToLLVMIR( mod, llvmContext, llvmSourceFilename );
        }

        if ( !llvmModule )
        {
            // coverage: bad-llvm-ir.ll
            llvm::errs() << COMPILER_NAME ": error: Failed to translate to LLVM IR or parse supplied LLVM IR\n";
            return ReturnCodes::loweringError;
        }

        // Verify the module to ensure debug info is valid
        if ( llvm::verifyModule( *llvmModule, &llvm::errs() ) )
        {
            // TODO: no coverage
            llvm::errs() << COMPILER_NAME ": error: Invalid LLVM IR module\n";
            return ReturnCodes::loweringError;
        }

        return ReturnCodes::success;
    }

    ReturnCodes CompilationUnit::runOptimizationPasses()
    {
        std::string targetTripleStr = llvm::sys::getProcessTriple();
        llvm::Triple targetTriple( targetTripleStr );
        llvmModule->setTargetTriple( targetTriple );

        // Lookup the target
        std::string error;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget( targetTriple, error );
        if ( !target )
        {
            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to find target: {}\n", error );
            return ReturnCodes::loweringError;
        }

        // Create the target machine
        targetMachine.reset(
            target->createTargetMachine( targetTriple, "generic", "", llvm::TargetOptions(), std::nullopt ) );

        if ( !targetMachine )
        {
            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create target machine\n" );
            return ReturnCodes::loweringError;
        }

        // Claude:
        //   At O0, buildPerModuleDefaultPipeline returns a nearly empty pass pipeline — it's essentially a no-op in terms
        //   of optimization. However it's not completely inert; it still runs a few mandatory passes:
        //
        //   * Annotation-to-metadata lowering — converts LLVM IR annotations to metadata, required for correctness.
        //   * CoroEarly / CoroCleanup — coroutine lowering passes, which are structural rather than optimizing.
        //   * Verifier — validates the IR is well-formed."
        //
        // Omitted entirely at O0 to see what differences are observable in the serialized LL:
        if ( ds.opt != llvm::OptimizationLevel::O0 )
        {
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

        return ReturnCodes::success;
    }

    InputType CompilationUnit::getInputType( llvm::StringRef filename )
    {
        llvm::StringRef ext = llvm::sys::path::extension( filename );

        if ( ext == ".mlir" || ext == ".sir" )
        {
            return InputType::MLIR;
        }

        if ( ext == ".ll" )
        {
            return InputType::LLVMLL;
        }

        if ( ext == ".mlirbc" || ext == ".sirbc" )
        {
            return InputType::MLIRBC;
        }

        if ( ext == ".bc" )
        {
            return InputType::LLVMBC;
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

    ReturnCodes CompilationUnit::serializeModuleMLIR( const llvm::SmallString<128>& mlirOutputName )
    {
        if ( !rmod )
        {
            return ReturnCodes::success;
        }

        if ( ds.emitMLIR or ds.emitMLIRBC )
        {
            std::error_code EC;
            llvm::raw_fd_ostream out( mlirOutputName.str(), EC,
                                      ds.emitMLIRBC ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text );
            if ( EC )
            {
                // coverage: bad-mlir-output-path-should-fail.silly
                llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file {}: {}\n",
                                             std::string( mlirOutputName ), EC.message() );
                return ReturnCodes::openError;
            }

            if ( ds.emitMLIRBC )
            {
                if ( mlir::failed( mlir::writeBytecodeToFile( *rmod, out ) ) )
                {
                    // TODO: no coverage.  Trigger with quotas or small filesystem?
                    llvm::errs() << std::format( COMPILER_NAME ": error: Failed to write bytecode to '{}'\n",
                                                 std::string( mlirOutputName ) );
                    return ReturnCodes::ioError;
                }
            }
            else
            {
                (*rmod).print( out, flags );
                out.close();
                if ( out.has_error() )
                {
                    // TODO: no coverage.  Trigger with quotas or small filesystem?
                    llvm::errs() << std::format( COMPILER_NAME ": error: Write error on '{}': {}\n",
                                                 std::string( mlirOutputName ), out.error().message() );
                    return ReturnCodes::ioError;
                }
            }
        }

        return ReturnCodes::success;
    }

    ReturnCodes CompilationUnit::serializeModuleLLVMIR( const llvm::SmallString<128>& llvmOuputFile )
    {
        // Dump the pre-optimized LL if we aren't creating a .o
        if ( !( ds.emitLLVM || ds.emitLLVMBC ) )
        {
            return ReturnCodes::success;
        }

        std::error_code EC;
        llvm::raw_fd_ostream out( llvmOuputFile.str(), EC,
                                  ds.emitLLVMBC ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text );
        if ( EC )
        {
            // coverage: bad-llvm-ir-output-path-should-fail.silly
            // FIXME: probably want llvm::formatv here and elsewhere to avoid the std::string casting hack (assuming
            // it knows how to deal with StringRef)
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open file '{}': {}\n",
                                         std::string( llvmOuputFile ), EC.message() );
            return ReturnCodes::openError;
        }

        if ( ds.emitLLVMBC )
        {
            llvm::WriteBitcodeToFile( *llvmModule, out );
        }
        else
        {
            llvmModule->print( out, nullptr, ds.debugInfo /* print debug info */ );
        }

        out.close();
        if ( out.has_error() )
        {
            // TODO: no coverage.  Trigger with quotas or small filesystem?
            llvm::errs() << std::format( COMPILER_NAME ": error: Write error on '{}': {}\n",
                                         std::string( llvmOuputFile ), out.error().message() );
            return ReturnCodes::openError;
        }

        return ReturnCodes::success;
    }

    ReturnCodes CompilationUnit::serializeObjectCode( const llvm::SmallString<128>& outputFilename )
    {
        std::error_code EC;
        llvm::raw_fd_ostream dest( outputFilename.str(), EC, llvm::sys::fs::OF_None );
        if ( EC )
        {
            // coverage: bad-object-output-path-should-fail.silly
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to open output file '{}': {}\n",
                                         std::string( outputFilename ), EC.message() );
            return ReturnCodes::openError;
        }

        llvmModule->setDataLayout( targetMachine->createDataLayout() );
        llvm::legacy::PassManager codegenPM;
        if ( targetMachine->addPassesToEmitFile( codegenPM, dest, nullptr, llvm::CodeGenFileType::ObjectFile ) )
        {
            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: TargetMachine can't emit an object file\n" );
            return ReturnCodes::loweringError;
        }

        codegenPM.run( *llvmModule );
        dest.close();

        LLVM_DEBUG( { llvm::outs() << "Generated object file: " << outputFilename << '\n'; } );

        return ReturnCodes::success;
    }

    ReturnCodes CompilationUnit::parseMLIRFile( const std::string& mlirSourceName )
    {
        auto fileOrErr = llvm::MemoryBuffer::getFile( mlirSourceName );
        if ( std::error_code EC = fileOrErr.getError() )
        {
            // coverage: bad-mlir-path-should-fail.silly
            llvm::errs() << std::format( COMPILER_NAME ": error: Cannot open file '{}': {}\n", mlirSourceName,
                                         EC.message() );
            return ReturnCodes::openError;
        }

        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer( std::move( *fileOrErr ), llvm::SMLoc{} );

        rmod = mlir::parseSourceFile<mlir::ModuleOp>( sourceMgr, context );
        if ( !rmod )
        {
            // coverage: bad-mlir-should-fail.mlir
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to parse MLIR file '{}'\n", mlirSourceName );
            return ReturnCodes::parseError;
        }

        return ReturnCodes::success;
    }

    ReturnCodes CompilationUnit::parseLLVMFile( const std::string& llvmSourceName )
    {
        llvm::SMDiagnostic err;
        llvmModule = llvm::parseIRFile( llvmSourceName, err, llvmContext );
        if ( !llvmModule )
        {
            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to parse IR file '{}': {}\n", llvmSourceName,
                                         err.getMessage().str() );
            return ReturnCodes::parseError;
        }

        return ReturnCodes::success;
    }
}    // namespace silly
