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
    void fatalDriverError( ReturnCodes rc );

    static void showLinkCommand( const std::string& linker, llvm::SmallVector<llvm::StringRef, 24>& args )
    {
        llvm::errs() << "# " << linker;

        for ( const auto& a : args )
        {
            llvm::errs() << a << ' ';
        }

        llvm::errs() << '\n';
    }

    SourceManager::SourceManager( silly::DriverState& d, mlir::MLIRContext* c, const std::string& firstFile )
        : ds{ d }, context{ c }
    {
        llvm::StringRef dirname = llvm::sys::path::parent_path( firstFile );
        // Create output directory if specified
        if ( !ds.outDir.empty() )
        {
            std::error_code EC = llvm::sys::fs::create_directories( ds.outDir );
            if ( EC )
            {
                llvm::errs() << std::format( COMPILER_NAME ": error: Failed to create output directory '{}': {}\n",
                                             ds.outDir, EC.message() );
                silly::fatalDriverError( ReturnCodes::directoryError );
            }

            outdir = ds.outDir;
        }
        else if ( dirname != "" )
        {
            outdir = dirname;
        }

        llvm::StringRef stem = llvm::sys::path::stem( firstFile );    // foo/bar.silly -> stem: is just bar, not foo/bar
        if ( stem.empty() )
        {
            llvm::errs() << std::format( COMPILER_NAME ": error: Invalid filename '{}', empty stem\n", firstFile );
            silly::fatalDriverError( ReturnCodes::filenameParseError );
        }

        defaultExecutablePath = outdir;
        if ( defaultExecutablePath.empty() )
        {
            defaultExecutablePath = stem;
        }
        else
        {
            llvm::sys::path::append( defaultExecutablePath, stem );
        }

        if ( ds.oName.empty() )
        {
            // This exe-path should be split out from CompilationUnit, as it may not match the input source file
            // stem. The defaultExecutablePath stuff is convoluted and confusing.
            exeName = defaultExecutablePath;
        }
        else
        {
            exeName = ds.oName;
        }
    }

    SourceManager::FileNameAndCU& SourceManager::createCU( const std::string& filename )
    {
        std::string stem = llvm::sys::path::stem( filename ).str();

        auto i = CUs.find( stem );
        if ( i != CUs.end() )
        {
            llvm::errs() << std::format( COMPILER_NAME ": error: file stem {} specified multiple times\n", stem );
            silly::fatalDriverError( ReturnCodes::duplicateCUError );
        }

        CUs[stem] = FileNameAndCU( filename, new silly::CompilationUnit( *this ) );

        return CUs[stem];
    }

    SourceManager::FileNameAndCU& SourceManager::findCU( const std::string& filename )
    {
        std::string stem = llvm::sys::path::stem( filename ).str();

        auto i = CUs.find( stem );
        if ( i == CUs.end() )
        {
            llvm::errs() << std::format( COMPILER_NAME ": error: Failed to find CU for stem {}\n", stem );
            silly::fatalDriverError( ReturnCodes::missingCUError );
        }

        return i->second;
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

    void SourceManager::link()
    {
        if ( ds.compileOnly or ds.assembleOnly )
        {
            return;
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

            llvm::errs() << std::format( COMPILER_NAME ": error: Linker failed with exit code: {}, rc = {}\n", errMsg,
                                         result );
            silly::fatalDriverError( ReturnCodes::linkError );
        }
    }

    void SourceManager::createAndSerializeMLIR( FileNameAndCU& cup )
    {
        std::string& filename = cup.filename;
        auto cu = cup.pCU;

        cu->processSourceFile( filename );

        llvm::SmallString<128> mlirOutputPath = defaultExecutablePath;
        if ( ds.emitMLIRBC )
        {
            mlirOutputPath += ".mlirbc";
        }
        else
        {
            mlirOutputPath += ".mlir";
        }

        cu->serializeModuleMLIR( mlirOutputPath );
    }

    // return true if LLVM-IR was created (i.e.: more to do)
    bool SourceManager::createAndSerializeLLVM( FileNameAndCU& cup )
    {
        auto cu = cup.pCU;

        // ds.assembleOnly produces the mlir
        // ds.emitMLIRBC and ds.compileOnly produces the mlirbc
        //
        // -- we are done in either case.
        if ( ds.assembleOnly or ( ds.emitMLIRBC and ds.compileOnly ) )
        {
            return false;
        }

        if ( cu->getInputType() == silly::InputType::OBJECT )
        {
            objFiles.push_back( cup.filename );

            return false;
        }
        else
        {
            cu->mlirToLLVM( cup.filename );

            llvm::SmallString<128> llvmOuputFile = defaultExecutablePath;
            llvmOuputFile += ".ll";
            cu->serializeModuleLLVMIR( llvmOuputFile );

            cu->runOptimizationPasses();

            return true;
        }
    }

    void SourceManager::serializeObject( FileNameAndCU& cup )
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
                // FIXME: another place to use formatv
                llvm::errs() << std::format( COMPILER_NAME
                                             ": error: Failed to create temporary object file in path '{}': {}\n",
                                             std::string( p ), EC.message() );

                silly::fatalDriverError( ReturnCodes::tempCreationError );
            }

            if ( ds.keepTemps )
            {
                // FIXME: another place to use formatv
                llvm::errs() << std::format( COMPILER_NAME ": info: created temporary: {}\n",
                                             std::string( objectFilename ) );
            }

            createdTemporary = true;
        }


        cu->serializeObjectCode( objectFilename );

        objFiles.push_back( std::string( objectFilename ) );

        if ( createdTemporary )
        {
            tmpToDelete.push_back( std::string( objectFilename ) );
        }
    }
}    // namespace silly
