#! /usr/bin/bash

set -x

for i in \
error_redeclare.toy \
error_unassigned.toy \
error_undeclare.toy \
error_invalid_binary.toy \
error_invalid_unary.toy \
    ; do
    ../build/toycalculator --location $i --emit-mlir --emit-llvm --no-emit-object --output-directory out
    cat out/$i.mlir
done

