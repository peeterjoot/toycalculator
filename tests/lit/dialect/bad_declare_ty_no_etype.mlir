// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
      // CHECK: error: expected non-function type
      // CHECK: error: Failed to parse element type
      %0 = "silly.declare"() : () -> !silly.var<j32>
      %1 = arith.constant 0 : i32
      "func.return"(%1) : (i32) -> ()
  }
}
