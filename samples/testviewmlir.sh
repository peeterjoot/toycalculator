#! /usr/bin/bash

set -x

rm -rf out

for i in \
empty.toy \
simplest.toy \
dcl.toy \
foo.toy \
test.toy \
    ; do

    echo "### $i"
    cat $i
    ../build/toycalculator -g $i --emit-mlir --emit-llvm --no-emit-object --output-directory out
    cat out/${i%.*}.mlir
done


