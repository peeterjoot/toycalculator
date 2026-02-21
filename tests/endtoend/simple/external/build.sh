set -x

rm -f callext *.ll *.mlir *.o

#silly -c callext.silly  -g --emit-llvm --emit-mlir
#silly -c external2.silly  -g --emit-llvm --emit-mlir
#TOP=`git rev-parse --show-toplevel`
#gcc -g -o callext callext.o external2.o -L ${TOP}/build/lib -l silly_runtime -Wl,-rpath,${TOP}/build/lib

silly callext.silly external2.silly
