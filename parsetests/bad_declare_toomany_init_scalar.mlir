// RUN: not mlir-opt-silly --source %s 2>&1 | FileCheck %s

module {
  // CHECK: error: fixme.
  %c1_i64 = arith.constant 1 : i64
  %0 = "silly.declare"(%c1_i64, %c1_i64) <{sym_name = "x"}> : (i64, i64) -> !silly.var<i64>
}
