# Define the ANTLR generated source files
set(ANTLR_SOURCES
  ${ANTLR_OUTPUT_DIR}/SillyBaseListener.cpp
  ${ANTLR_OUTPUT_DIR}/SillyListener.cpp
  ${ANTLR_OUTPUT_DIR}/SillyParser.cpp
  ${ANTLR_OUTPUT_DIR}/SillyLexer.cpp
)

set(DRIVER_SOURCES
  CompilationUnit.cpp
  DialectContext.cpp
  driver.cpp
  helper.cpp
  LoweringContext.cpp
  lowering.cpp
  MlirTypeCache.cpp
  SourceManager.cpp
)

set_source_files_properties(
  ${DRIVER_SOURCES}
  PROPERTIES COMPILE_OPTIONS "-fno-rtti"
)

# Mark them as generated so CMake doesn't complain they don't exist yet
set_source_files_properties(
  ${ANTLR_SOURCES}
  PROPERTIES GENERATED TRUE
)

set_source_files_properties(
  ${ANTLR_SOURCES}
  Antlr4ParseListener.cpp
  PROPERTIES COMPILE_OPTIONS "-frtti"
)

add_compile_definitions(
    LLVMVERSION=\"${LLVM_PREFIX}\"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Try to read /etc/os-release (standard on modern distros)
    file(READ "/etc/os-release" os_release)
    string(REGEX MATCH "ID=([a-z]+)" _ "${os_release}")
    set(DISTRO_ID "${CMAKE_MATCH_1}")

    if(DISTRO_ID STREQUAL "ubuntu" OR DISTRO_ID STREQUAL "debian")
        # Ubuntu/Debian default to PIE linking → force PIC objects
        message(STATUS "Detected ${DISTRO_ID} → forcing PIC/PIE compatible code")
        add_compile_definitions(
            UBUNTU_PIC_REQUIRED=1
        )
    endif()
endif()

add_executable(silly
  ${ANTLR_SOURCES}
  ${DRIVER_SOURCES}
  Antlr4ParseListener.cpp
)

# Experiment to see what depends on RTTI (more than Antlr4ParseListener.cpp?): answer:
#FAILED: src/driver/CMakeFiles/silly.dir/__/grammar/SillyParser/SillyListener.cpp.o
#FAILED: src/driver/CMakeFiles/silly.dir/__/grammar/SillyParser/SillyBaseListener.cpp.o
#FAILED: src/driver/CMakeFiles/silly.dir/__/grammar/SillyParser/SillyLexer.cpp.o
#FAILED: src/driver/CMakeFiles/silly.dir/__/grammar/SillyParser/SillyParser.cpp.o
#FAILED: src/driver/CMakeFiles/silly.dir/Antlr4ParseListener.cpp.o
#target_compile_options(silly PRIVATE -fno-rtti)

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
  LLVMBitWriter
  LLVMIRReader
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
  MLIRTestIR
)

# Create bin directory and symlink
add_custom_command(TARGET silly POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_BINARY_DIR}/silly
        ${CMAKE_BINARY_DIR}/bin/silly
    COMMENT "Creating symlink: ${CMAKE_BINARY_DIR}/bin/silly -> ${CMAKE_CURRENT_BINARY_DIR}/silly"
)

