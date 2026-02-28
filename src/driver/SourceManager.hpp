/// @file    SourceManager.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler handling of a set of sources or objects.
#pragma once

#include <vector>
#include <unordered_map>

namespace silly
{
    class CompilationUnit;

    /// State associated with the compilation workflow of one or more sources.
    class SourceManager
    {
       public:
        /// Create the directory named in --output-directory.
        ///
        /// Populates outdir as a side effect with the output directory, or the directory
        /// part of the filename path (if specified), or an empty string.
        SourceManager( silly::DriverState& d, mlir::MLIRContext* c, const std::string& firstFile );

        ~SourceManager();

        /// A filename and CompilationUnit (pointer) pair.
        struct FileNameAndCU
        {
            std::string filename{}; ///< indexing the map by stem.
            silly::CompilationUnit* pCU{}; ///< compilation state for a single named source file.

            /// specify a filename and raw-pointer to a CompilationUnit.
            FileNameAndCU( const std::string & f = "", silly::CompilationUnit* p = nullptr ) : filename{f}, pCU{p} {}
        };

        /// Create a CompilationUnit for this file
        FileNameAndCU& createCU( const std::string& filename );

        /// Find and return the CompilationUnit for this file
        FileNameAndCU& findCU( const std::string& filename );

        /// Extract the stem from filename and look for a compiled module for it.
        mlir::ModuleOp findMOD( const std::string& filename );

        /// Build the MLIR representation of the named source, if required, and emit it (if desired.)
        void createAndSerializeMLIR( FileNameAndCU& cup );

        /// Lower the MLIR module to LLVM-IR (and emit that as a file if desired.)
        /// @retval true if LLVM-IR was created (i.e.: more to do)
        bool createAndSerializeLLVM( FileNameAndCU& cup );

        /// Write out the object file to disk for the linker (if desired.)
        void serializeObject( FileNameAndCU& cup );

        /// Take the current filename, grab the stem, and add the suffix.  This includes the --output-directory if specified.
        void constructPathForStem( llvm::SmallString<128> & outputPath, const std::string & sourceName, const char * suffixWithDot );

        /// Invoke the system linker to create an executable.
        void link();

        /// Getter for DriverState
        silly::DriverState& getDriverState()
        {
            return ds;
        }

        /// Getter for the mlir::MLIRContext
        mlir::MLIRContext* getContext()
        {
            return context;
        }

       private:
        /// Back reference to the driver state
        silly::DriverState& ds;

        /// Back pointer to the owning context (for all the modules to be created.)
        mlir::MLIRContext* context;

        /// A map of FileNameAndCUs, indexed by filename stem.
        std::unordered_map<std::string, FileNameAndCU> CUs{};

        /// The executable path/name to be generated.
        llvm::SmallString<128> exeName;

        /// Just the output directory, based on the filename of the first source and --output-directory
        llvm::SmallString<128> outdir;

        /// The set of object files to be linked.
        std::vector<std::string> objFiles;

        /// All the temporary object files in $TMPDIR to be cleaned up on exit.
        std::vector<std::string> tmpToDelete;

        /// Output directory combined with filename stem (no extension)
        llvm::SmallString<128> defaultExecutablePath;
    };
}    // namespace silly
