rm -rf build
mkdir build
cd build
cmake -G Ninja \
      -DLLVM_DIR=/usr/lib/llvm-19/cmake \
      -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
      ..

