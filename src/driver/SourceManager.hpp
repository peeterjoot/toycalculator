/// @file    SourceManager.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler handling of a set of sources or objects.
#pragma once

#include <vector>
#include <unordered_map>

namespace silly
{
    class CompilationUnit;

    class SourceManager
    {
       public:
        /// Create the directory named in --output-directory.
        ///
        /// Populates outdir as a side effect with the output directory, or the directory
        /// part of the filename path (if specified), or an empty string.
        SourceManager( silly::DriverState& d, mlir::MLIRContext* c, const std::string& firstFile );

        ~SourceManager();

        struct FileNameAndCU
        {
            std::string filename{}; ///< indexing the map by stem.
            silly::CompilationUnit* pCU{};

            FileNameAndCU( const std::string & f = "", silly::CompilationUnit* p = nullptr ) : filename{f}, pCU{p} {}
        };

        FileNameAndCU& createCU( const std::string& filename );

        FileNameAndCU& findCU( const std::string& filename );

        /// Extract the stem from filename and look for a compiled module for it.
        mlir::ModuleOp findMOD( const std::string& filename );

        void createAndSerializeMLIR( FileNameAndCU& cup );

        bool createAndSerializeLLVM( FileNameAndCU& cup );

        void serializeObject( FileNameAndCU& cup );

        /// Invoke the system linker to create an executable.
        void link();

        silly::DriverState& getDriverState()
        {
            return ds;
        }

        mlir::MLIRContext* getContext()
        {
            return context;
        }

       private:
        silly::DriverState& ds;
        mlir::MLIRContext* context;
        std::unordered_map<std::string, FileNameAndCU> CUs{};

        llvm::SmallString<128> exeName;

        /// Just the output directory, based on the filename of the first source and --output-directory
        llvm::SmallString<128> outdir;

        std::vector<std::string> objFiles;
        std::vector<std::string> tmpToDelete;

        /// Output directory combined with filename stem (no extension)
        llvm::SmallString<128> defaultExecutablePath;
    };
}    // namespace silly
