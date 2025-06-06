#! /usr/bin/bash

set -x

rm -rf out
mkdir out

for i in \
error_redeclare.toy \
error_unassigned.toy \
error_undeclare.toy \
error_invalid_binary.toy \
error_invalid_unary.toy \
error_keyword_declare.toy \
error_keyword_declare2.toy \
    ; do
    ../build/toycalculator -g $i --emit-mlir --emit-llvm --no-emit-object --output-directory out > out/$i.out 2>&1
    rc=$?
    echo "RC = $rc"

    if [ $rc -eq 0 ] ; then
        echo "unexpected success compiling '$i'"
        exit 1
    fi
    #cat out/$i.mlir
done

echo "testerrors: ALL TESTS UNSUCCESSFUL, AS EXPECTED"

# vim: et ts=4 sw=4
