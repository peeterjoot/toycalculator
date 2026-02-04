#------------------------------------------------------------------------------
# mostly tablegen rules:
#

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen for dialect
set(LLVM_TARGET_DEFINITIONS silly.td)

# Generate dialect declarations
mlir_tablegen(SillyDialectDecls.hpp.inc -gen-dialect-decls -dialect silly)

# Generate dialect definitions
mlir_tablegen(SillyDialectDefs.cpp.inc -gen-dialect-defs -dialect silly)

# Generate operation declarations
mlir_tablegen(SillyDialect.hpp.inc -gen-op-decls -dialect silly)

# Generate operation definitions
mlir_tablegen(SillyDialect.cpp.inc -gen-op-defs -dialect silly)

# Generate type declarations
mlir_tablegen(
  SillyTypes.hpp.inc
  -gen-typedef-decls
  --typedefs-dialect=silly
)

# Generate type definitions
mlir_tablegen(
  SillyTypes.cpp.inc
  -gen-typedef-defs
  --typedefs-dialect=silly
)

# Create a single TableGen target for all generated the base dialect files
add_public_tablegen_target(SillyDialectIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/SillyDialectDecls.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/SillyDialectDefs.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/SillyDialect.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/SillyDialect.cpp.inc
)

# And a TableGen target for the types files
add_public_tablegen_target(SillyTypesIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/SillyTypes.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/SillyTypes.cpp.inc
)

# and one for the MLIR -> LLVM-IR lowering TableGen pass
set(LLVM_TARGET_DEFINITIONS SillyPasses.td)

#mlir_tablegen(SillyPasses.hpp.inc --gen-pass-decls -name Silly -I ${CMAKE_SOURCE_DIR}/src -I ${MLIR_INCLUDE_DIRS})
mlir_tablegen(SillyPasses.hpp.inc --gen-pass-decls -name Silly -I ${MLIR_INCLUDE_DIRS})

add_public_tablegen_target(SillyPassesIncGen
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/SillyPasses.hpp.inc
)

add_library(SillyDialect SHARED
  SillyDialect.cpp
  SillyTypes.cpp
  SillyPasses.cpp
)

#------------------------------------------------------------------------------

#add_dependencies(SillyDialect GenerateANTLR SillyDialectIncGen SillyPassesIncGen SillyTypesIncGen)
add_dependencies(SillyDialect SillyDialectIncGen SillyPassesIncGen SillyTypesIncGen)

target_link_libraries(SillyDialect
  PUBLIC
    MLIRIR
    MLIRSupport
)

# Create symlink in build/lib/
add_custom_command(TARGET SillyDialect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ../src/dialect/libSillyDialect.so
        ${CMAKE_BINARY_DIR}/lib/libSillyDialect.so
    COMMENT "Creating symlink: ${CMAKE_BINARY_DIR}/lib/libSillyDialect.so -> src/dialect/libSillyDialect.so"
)

if(SILLY_ENABLE_INSTALL)
    # Set RPATH for installed binary
    set_target_properties(SillyDialect PROPERTIES
        INSTALL_RPATH "${LLVM_PREFIX}/lib64"
    )

    install(TARGETS SillyDialect
        LIBRARY DESTINATION lib
        NAMELINK_SKIP
    )
endif()
