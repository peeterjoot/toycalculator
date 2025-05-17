#! /usr/bin/bash

flags='-O 2'

rm -rf out

#test.toy # need binaryop lowering first.
for i in \
    addi.toy \
    test.toy \
    unary.toy \
    bool.toy \
    exit3.toy \
    exitx.toy \
    types.toy \
    empty.toy \
    simplest.toy \
    dcl.toy \
    foo.toy \
    bin.toy \
; do

    stem=${i%.*}
    echo "##########################################################################"
    echo "// $i"
    cat $i
    echo "##########################################################################"
    echo "../build/toycalculator --output-directory out -g $i $flags --emit-llvm --emit-mlir"
    ../build/toycalculator --output-directory out -g $i $flags --emit-llvm --emit-mlir

    echo objdump -dr out/${stem}.o
    objdump -dr out/${stem}.o

    echo clang -g -o out/${stem} out/${stem}.o -L ../build -l toy_runtime -Wl,-rpath,`pwd`/../build
    clang -g -o out/${stem} out/${stem}.o -L ../build -l toy_runtime -Wl,-rpath,`pwd`/../build

    echo ./out/${stem}
    ./out/${stem}

    echo "RC = " $?
    exit
done

# ../build/toycalculator empty.toy -g --stdout --no-emit-object --emit-mlir --emit-llvm --debug

# vim: et ts=4 sw=4
