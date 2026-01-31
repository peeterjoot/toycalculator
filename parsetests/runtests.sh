#! /usr/bin/bash

MLIROPT=/usr/local/llvm-21.1.8/bin/mlir-opt

set -x

#${MLIROPT} \
#--load-dialect-plugin=../build/libSillyDialect.so \
#  out/loadstore.mlir \
#  -o -

# nice easy to read output:
#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so \
#         -canonicalize \
#         -cse \
#         -symbol-dce \
#         out/loadstore.mlir -o - | less


# Print in generic mode (to see internal representation):
#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so \
#         -mlir-print-op-generic \
#         out/loadstore.mlir -o - | less

# round trip:
#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so \
#         out/loadstore.mlir \
#         | ${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so \
#         -o roundtrip.mlir

#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so bad1.mlir
#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so bad2.mlir
#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so bad3.mlir
#${MLIROPT} --load-dialect-plugin=../build/libSillyDialect.so good1.mlir
