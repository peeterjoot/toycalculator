# Define the ANTLR generated source files
set(ANTLR_SOURCES
  ${ANTLR_OUTPUT_DIR}/SillyBaseListener.cpp
  ${ANTLR_OUTPUT_DIR}/SillyListener.cpp
  ${ANTLR_OUTPUT_DIR}/SillyParser.cpp
  ${ANTLR_OUTPUT_DIR}/SillyLexer.cpp
)

# Mark them as generated so CMake doesn't complain they don't exist yet
set_source_files_properties(
  ${ANTLR_SOURCES}
  PROPERTIES GENERATED TRUE
)

add_executable(silly
  ${DRIVER_SOURCE_FILES}
  driver.cpp
  lowering.cpp
  parser.cpp
  helper.cpp
  LoweringContext.cpp
  ${ANTLR_SOURCES}
)

if(SILLY_ENABLE_INSTALL)
    # Set RPATH for installed binary
    set_target_properties(silly PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib;${LLVM_PREFIX}/lib64"
    )

    install(TARGETS silly
        RUNTIME DESTINATION bin
    )
endif()

# Include the output directory for header files
target_include_directories(silly PRIVATE ${ANTLR_OUTPUT_DIR})

# Ensure the ANTLR and TableGen generation runs before building the project
add_dependencies(silly silly_runtime SillyDialect GenerateANTLR)

target_compile_options(silly PRIVATE ${COMMON_COMPILE_OPTIONS})

# Check the target architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(IS_AARCH64 TRUE)
    set(LLVM_TARGET_ARCH "AArch64")
else()
    set(IS_AARCH64 FALSE)
    set(LLVM_TARGET_ARCH "X86")
endif()

# Optional: Print the results
message(STATUS "Architecture is aarch64: ${IS_AARCH64}")
message(STATUS "LLVM Architecture specific libraries: ${LLVM_ARCH_LIBS}")

target_link_libraries(silly
  PRIVATE
  antlr4-runtime
  SillyDialect
  LLVMSupport
  LLVMMC # llvm::TargetRegistry::lookupTarget
  LLVMAnalysis # llvm::AnalysisManager<llvm::LazyCallGraph::SCC, llvm::LazyCallGraph&>::AnalysisManager()
  LLVMPasses
  LLVM${LLVM_TARGET_ARCH}AsmParser
  LLVM${LLVM_TARGET_ARCH}CodeGen
  LLVM${LLVM_TARGET_ARCH}Desc
  LLVM${LLVM_TARGET_ARCH}Info
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRAsmParser
  MLIRTargetLLVMIRExport    # mlir::translateModuleToLLVMIR
  MLIRDataLayoutInterfaces
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
)

# Create bin directory and symlink
add_custom_command(TARGET silly POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_BINARY_DIR}/silly
        ${CMAKE_BINARY_DIR}/bin/silly
    COMMENT "Creating symlink: ${CMAKE_BINARY_DIR}/bin/silly -> ${CMAKE_CURRENT_BINARY_DIR}/silly"
)

