// RUN: not mlir-opt-silly --source %s 2>&1 | FileCheck %s

module {
  // CHECK: error: 'silly.declare' op requires attribute 'sym_name'
  %v = "silly.declare"() : () -> !silly.var<i64[1]>
}
