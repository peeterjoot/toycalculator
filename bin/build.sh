set -x

rm -rf build
mkdir build
cd build

# didn't pick consistent install prefixes for my mlir builds on the two machines:

if [ -d /usr/local/llvm-20 ] ; then
   PREFIX=/usr/local/llvm-20
fi
if [ -d /usr/local/llvm-20.1.2 ] ; then
   PREFIX=/usr/local/llvm-20.1.2
fi
if [ -d /usr/local/llvm-20.1.3 ] ; then
   PREFIX=/usr/local/llvm-20.1.3
fi

cmake -G Ninja \
-DLLVM_DIR=$PREFIX/lib64/cmake/llvm \
-DMLIR_DIR=$PREFIX/lib64/cmake/mlir \
..

ninja

my_cscope --build
