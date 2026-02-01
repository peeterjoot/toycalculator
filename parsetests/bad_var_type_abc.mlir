// RUN: not mlir-opt-silly --source %s 2>&1 | FileCheck %s

module {
  // CHECK: error: expected integer value
  %bad = "silly.declare"() : () -> !silly.var<i64[abc]>
}
