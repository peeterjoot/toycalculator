#! /usr/bin/bash

set -x

#test.toy # need binaryop lowering first.
for i in \
empty.toy \
simplest.toy \
dcl.toy \
foo.toy \
    ; do

    echo "### $i"
    ../build/toycalculator $i --output-directory out --location
    objdump -dr out/${i%.*}.o
    clang -o out/${i%.*} out/${i%.*}.o -L ../build -l toy_runtime -Wl,-rpath,`pwd`/../build
    ./out/${i%.*}
done

# vim: et ts=4 sw=4
