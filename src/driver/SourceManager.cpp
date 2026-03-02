/// @file    SourceManager.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler handling of a set of sources or objects.
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include <format>
#include <fstream>

#include "CompilationUnit.hpp"
#include "ParseListener.hpp"
#include "ReturnCodes.hpp"
#include "SourceManager.hpp"

#define DEBUG_TYPE "silly-sm"

namespace silly
{
    static void showLinkCommand( const std::string& linker, llvm::SmallVector<llvm::StringRef, 24>& args )
    {
        // TODO: no coverage
        llvm::errs() << "# " << linker;

        for ( const auto& a : args )
        {
            llvm::errs() << a << ' ';
        }

        llvm::errs() << '\n';
    }

    SourceManager::SourceManager( silly::DriverState& d, mlir::MLIRContext* c )
        : ds{ d }, context{ c }
    {
    }

    ReturnCodes SourceManager::constructOutputDirectory( const std::string& firstFile )
    {
        llvm::StringRef dirname = llvm::sys::path::parent_path( firstFile );
        // Create output directory if specified
        if ( !ds.outDir.empty() )
        {
            std::error_code EC = llvm::sys::fs::create_directories( ds.outDir );
            if ( EC )
            {
                // TODO: no coverage
                llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create output directory '{}': {}\n",
                                             ds.outDir, EC.message() );
                return ReturnCodes::directoryError;
            }

            outdir = ds.outDir;
        }
        else if ( dirname != "" )
        {
            outdir = dirname;
        }

        constructPathForStem( defaultExecutablePath, firstFile, "" );
        exeName = defaultExecutablePath;

        return ReturnCodes::success;
    }

    ReturnCodes SourceManager::createCU( const std::string& filename, SourceManager::FileNameAndCU *& cup )
    {
        cup = nullptr;
        std::string stem = llvm::sys::path::stem( filename ).str();

        auto i = CUs.find( stem );
        if ( i != CUs.end() )
        {
            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: file stem {} specified multiple times\n", stem );
            return ReturnCodes::duplicateCUError;
        }

        CUs[stem] = FileNameAndCU( filename, new silly::CompilationUnit( *this ) );

        cup = &CUs[stem];

        return ReturnCodes::success;
    }

    ReturnCodes SourceManager::findCU( const std::string& filename, SourceManager::FileNameAndCU *& cup )
    {
        cup = nullptr;
        std::string stem = llvm::sys::path::stem( filename ).str();
        auto i = CUs.find( stem );
        if ( i == CUs.end() )
        {
            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to find CU for stem {}\n", stem );
            return ReturnCodes::missingCUError;
        }

        cup = &i->second;

        return ReturnCodes::success;
    }

    mlir::ModuleOp SourceManager::findMOD( const std::string& filename )
    {
        std::string stem = llvm::sys::path::stem( filename ).str();

        auto i = CUs.find( stem );
        if ( i != CUs.end() )
        {
            assert( i->second.pCU );
            return i->second.pCU->getModule();
        }

        return nullptr;
    }

    SourceManager::~SourceManager()
    {
        for ( auto& i : CUs )
        {
            delete i.second.pCU;
        }

        if ( !ds.keepTemps )
        {
            for ( const auto& filename : tmpToDelete )
            {
                llvm::sys::fs::remove( filename );
            }
        }
    }

    ReturnCodes SourceManager::link()
    {
        if ( ds.compileOnly )
        {
            return ReturnCodes::success;
        }

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

            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: Error finding path for linker '{}': {}\n", linker,
                                         EC.message() );
            return ReturnCodes::filenameParseError;
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
        linkerArgValues.push_back( std::string( exeName ) );
        for ( const auto& o : objFiles )
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

        if ( ds.verboseLink )
        {
            showLinkCommand( linkerPath.get(), linkerArgs );
        }

        // Execute the linker
        std::string errMsg;
        int result = llvm::sys::ExecuteAndWait( linkerPath.get(), linkerArgs, std::nullopt, {}, 0, 0, &errMsg );
        if ( result != 0 )
        {
            if ( !ds.verboseLink )    // already showed this
            {
                showLinkCommand( linkerPath.get(), linkerArgs );
            }

            // TODO: no coverage
            llvm::errs() << std::format( COMPILER_NAME ": error: Linker failed with exit code: {}, rc = {}\n", errMsg,
                                         result );
            return ReturnCodes::linkError;
        }

        return ReturnCodes::success;
    }

    void SourceManager::constructPathForStem( llvm::SmallString<128>& outputPath, const std::string& sourceName,
                                              const char* suffixWithDot )
    {
        // FIXME: there is portable LLVM infra for path construction (but I don't currently care about Windows, so "/"
        // is okay for now)
        if ( !ds.oName.empty() )
        {
            // If outputPath is fully qualified, ignore any implicit (constructed from the path of the source) output
            // directory or any explicit --output-directory:
            if ( !outdir.empty() and ( ds.oName[0] != '/' ) )
            {
                outputPath = outdir;
                outputPath += "/";
            }

            // Now want to distinguish -o exename, and --emit-llvm/--emit-mlir (without -c) where
            // --emit-llvm/--emit-mlir is just to produce a supplementary listing, and let the
            // compile/lower/assemble/link proceed.
            if ( !ds.compileOnly and ( ds.emitLLVM or ds.emitLLVMBC or ds.emitMLIR or ds.emitMLIRBC ) and
                 suffixWithDot[0] )
            {
                llvm::StringRef fn = llvm::sys::path::stem( llvm::sys::path::filename( sourceName ) );
                outputPath += fn;
                outputPath += suffixWithDot;
            }
            else
            {
                outputPath += ds.oName;
            }
        }
        else
        {
            llvm::StringRef stem = llvm::sys::path::stem( sourceName );
            if ( !outdir.empty() )
            {
                outputPath = outdir;
                outputPath += "/";
                outputPath += stem;
            }
            else
            {
                outputPath += stem;
            }

            outputPath += suffixWithDot;
        }
    }

    ReturnCodes SourceManager::createAndSerializeMLIR( FileNameAndCU& cup )
    {
        std::string& filename = cup.filename;
        auto cu = cup.pCU;

        ReturnCodes rc = cu->processSourceFile( filename );
        if ( rc != ReturnCodes::success )
        {
            return rc;
        }

        llvm::SmallString<128> mlirOutputPath;
        constructPathForStem( mlirOutputPath, filename, ds.emitMLIRBC ? ".mlirbc" : ".mlir" );
        rc = cu->serializeModuleMLIR( mlirOutputPath );
        if ( rc != ReturnCodes::success )
        {
            return rc;
        }

        return ReturnCodes::success;
    }

    ReturnCodes SourceManager::createAndSerializeLLVM( FileNameAndCU& cup, bool & isDone )
    {
        auto cu = cup.pCU;
        isDone = false;

        // ds.emitMLIR and ds.compileOnly produces the mlir (only, no object)
        // ds.emitMLIRBC and ds.compileOnly produces the mlirbc (only, no object)
        //
        // -- we are done in either case.
        if ( ( ds.emitMLIRBC or ds.emitMLIR ) and ds.compileOnly )
        {
            return ReturnCodes::success;
        }

        if ( cu->getInputType() == silly::InputType::OBJECT )
        {
            objFiles.push_back( cup.filename );

            return ReturnCodes::success;
        }

        ReturnCodes rc = cu->mlirToLLVM( cup.filename );
        if ( rc != ReturnCodes::success )
        {
            return rc;
        }

        rc = cu->runOptimizationPasses();
        if ( rc != ReturnCodes::success )
        {
            return rc;
        }

        // Serialize only after any passes have been run.
        llvm::SmallString<128> llvmOutputPath;
        constructPathForStem( llvmOutputPath, cup.filename, ds.emitLLVMBC ? ".bc" : ".ll" );
        rc = cu->serializeModuleLLVMIR( llvmOutputPath );
        if ( rc != ReturnCodes::success )
        {
            return rc;
        }

        // -c --emit-llvm, or -c --emit-llvmbc
        if ( ds.compileOnly )
        {
            if ( ds.emitLLVM or ds.emitLLVMBC )
            {
                return ReturnCodes::success;
            }
        }

        isDone = true;

        return ReturnCodes::success;
    }

    ReturnCodes SourceManager::serializeObject( FileNameAndCU& cup )
    {
        llvm::SmallString<128> objectFilename;
        bool createdTemporary{};
        const std::string& filename = cup.filename;
        auto cu = cup.pCU;

        if ( !ds.oName.empty() && ds.compileOnly )
        {
            objectFilename = ds.oName;
        }
        else if ( ds.compileOnly )
        {
            objectFilename += defaultExecutablePath;
            objectFilename += ".o";
        }
        else
        {
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
                // TODO: no coverage
                // FIXME: another place to use formatv
                llvm::errs() << std::format( COMPILER_NAME
                                             ": error: Failed to create temporary object file in path '{}': {}\n",
                                             std::string( p ), EC.message() );

                return ReturnCodes::tempCreationError;
            }

            if ( ds.keepTemps )
            {
                // TODO: no coverage
                // FIXME: another place to use formatv
                llvm::errs() << std::format( COMPILER_NAME ": info: created temporary: {}\n",
                                             std::string( objectFilename ) );
            }

            createdTemporary = true;
        }


        ReturnCodes rc = cu->serializeObjectCode( objectFilename );
        if ( rc != ReturnCodes::success )
        {
            return rc;
        }

        objFiles.push_back( std::string( objectFilename ) );

        if ( createdTemporary )
        {
            tmpToDelete.push_back( std::string( objectFilename ) );
        }

        return ReturnCodes::success;
    }
}    // namespace silly
