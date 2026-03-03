// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
      // CHECK: error: unbalanced '<' character in pretty dialect name
      %0 = "silly.declare"() : () -> !silly.var<i32
      %1 = arith.constant 0 : i32
      "func.return"(%1) : (i32) -> ()
  }
}
