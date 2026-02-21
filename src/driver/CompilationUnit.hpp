/// @file    CompilationUnit.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler handling of a single compilation unit.
///
/// Provides per-source functionality for:
///
/// - running the antlr4 parse tree listener (w/ MLIR builder),
/// - running the LLVM-IR lowering pass, and
/// - running the assembly printer.
///
#pragma once

#include <llvm/IR/Module.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>

namespace mlir
{
    class MLIRContext;
}    // namespace mlir

namespace silly
{
    class DriverState;

    /// Supported source code file extensions.
    enum class InputType
    {
        Silly,     // .silly or other source
        MLIR,      // .mlir
        OBJECT,    // .o
        Unknown
    };

    /// Track everything related to a specific compilation unit through all phases of the compilation pipeline.
    ///
    /// This class manages the state and orchestrates the transformation of a single source file
    /// through parsing, MLIR generation, LLVM IR lowering, optimization, and object code emission.
    class CompilationUnit
    {
       public:
        /// Construct a new module state for a compilation unit.
        ///
        /// @param f The source filename to compile
        /// @param c The MLIR context for this compilation
        CompilationUnit( silly::DriverState& d, std::string f, mlir::MLIRContext* c );

        /// Parse the source file and build the MLIR module.
        ///
        /// Determines input type (.silly, .mlir, .o) and invokes the appropriate parser.
        /// Populates rmod with the resulting MLIR module.
        void processSourceFile();

        /// Lower the MLIR module to LLVM IR dialect.
        ///
        /// Runs the MLIR-to-LLVM lowering passes and translates the result to an llvm::Module.
        /// Populates llvmModule as a side effect.
        void mlirToLLVM();

        /// Run LLVM optimization passes on the lowered module.
        ///
        /// Creates a target machine and runs the optimization pipeline based on the -O level.
        /// Modifies llvmModule in place.
        void runOptimizationPasses();

        /// Emit object code (.o file) from the optimized LLVM module.
        ///
        /// @param outputFilename[out] Path where the object file will be written
        void serializeObjectCode( const llvm::SmallString<128>& outputFilename );

        /// Get the detected input file type.
        ///
        /// @return The input type (.silly, .mlir, .o, or unknown)
        InputType getInputType() const
        {
            return ity;
        }

        /// Construct the output path for the object file.
        ///
        /// @param outputFilename[out] Buffer to receive the constructed path
        void constructObjectPath( llvm::SmallString<128>& outputFilename );

        /// Return the executable path associated with this file.
        const llvm::SmallString<128>& getDefaultExecutablePath() const
        {
            return dirWithStem;
        }

        const llvm::SmallString<128>& getOutputDirectory() const
        {
            return outdir;
        }

       private:
        silly::DriverState& ds;

        /// Source filename being compiled
        std::string filename{};

        /// MLIR context for this compilation unit
        mlir::MLIRContext* context{};

        /// The MLIR module (either parsed or generated from source)
        mlir::OwningOpRef<mlir::ModuleOp> rmod{};

        /// Detected input file type (.silly, .mlir, or .o)
        InputType ity{};

        /// Flags controlling MLIR printing (for debug output)
        mlir::OpPrintingFlags flags;

        /// LLVM context - must persist for lifetime of llvmModule
        llvm::LLVMContext llvmContext;

        /// The LLVM IR module after lowering from MLIR
        std::unique_ptr<llvm::Module> llvmModule;

        /// Target machine for code generation and optimization
        std::unique_ptr<llvm::TargetMachine> targetMachine;

        /// Output directory combined with filename stem (no extension)
        llvm::SmallString<128> dirWithStem;

        /// Just the output directory
        llvm::SmallString<128> outdir;

        /// Determine the input type from a filename extension.
        ///
        /// @param filename The filename to examine
        /// @return The detected input type
        static InputType getInputType( llvm::StringRef filename );

        /// Serialize the MLIR module to a .mlir file (if --emit-mlir specified).
        void serializeModuleMLIR();

        /// Serialize the LLVM IR module to a .ll file (if --emit-llvm specified).
        void serializeModuleLLVMIR();

        /// Create the directory named in --output-directory.
        ///
        /// Populates outdir as a side effect with the output directory, or the directory
        /// part of the filename path (if specified), or an empty string.
        void makeOutputDirectory();

        /// Parse a .mlir file into the MLIR module.
        ///
        /// Saves the parsed module to rmod as a side effect.
        void parseMLIRFile();
    };
}    // namespace silly
