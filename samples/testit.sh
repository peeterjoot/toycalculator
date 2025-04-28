set -x

for i in \
empty.toy \
simplest.toy \
dcl.toy \
foo.toy \
test.toy \
error_redeclare.toy \
error_unassigned.toy \
error_undeclare.toy \
error_invalid_binary.toy \
error_invalid_unary.toy \
    ; do
    echo $i
    cat $i
    ../build/toycalculator --location $i --emit-mlir
    #../build/toycalculator --location $i --emit-llvm
done

../build/toycalculator foo.toy --output-directory out --emit-llvm --emit-mlir --location
clang -o foo out/foo.o -L ../build -l toy_runtime -Wl,-rpath,`pwd`/../build
./foo

# vim: et ts=4 sw=4
