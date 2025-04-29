#! /usr/bin/bash

set -x

#test.toy # need binaryop lowering first.
for i in \
empty.toy \
simplest.toy \
dcl.toy \
foo.toy \
    ; do

    stem=${i%.*}
    echo "### $i"
    ../build/toycalculator $i --output-directory out --location
    objdump -dr out/${stem}.o
    clang -g -o out/${stem} out/${stem}.o -L ../build -l toy_runtime -Wl,-rpath,`pwd`/../build
    ./out/${stem}
done

# ../build/toycalculator empty.toy --location --stdout --no-emit-object --emit-mlir --emit-llvm --debug

# vim: et ts=4 sw=4
