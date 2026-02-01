// RUN: not mlir-opt-silly --source %s 2>&1 | FileCheck %s

module {
  // CHECK: error: %c1_i64 undefined.
  %0 = "silly.declare"(%c1_i64) <{sym_name = "anInitializedScalar"}> : (i64) -> !silly.var<i64>
}
