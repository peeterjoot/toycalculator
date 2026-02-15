// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  // CHECK: error: 'silly.declare' op number of initializers (2) exceeds number of elements (1)
  %c1_i64 = arith.constant 1 : i64
  %0 = "silly.declare"(%c1_i64, %c1_i64) : (i64, i64) -> !silly.var<i64[1]>
}
