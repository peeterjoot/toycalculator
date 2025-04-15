set -x

rm -rf build
mkdir build
cd build

cmake -G Ninja \
-DLLVM_DIR=/usr/local/llvm-20/lib64/cmake/llvm \
-DMLIR_DIR=/usr/local/llvm-20/lib64/cmake/mlir \
..

ninja
