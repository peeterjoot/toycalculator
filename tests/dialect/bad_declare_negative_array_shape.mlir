// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
      // Was trying to trigger declare verifier error: 'array size must be a positive integer' -- this one is coming from the type parser.
      // CHECK: error: array size must be positive
      %0 = "silly.declare"() : () -> !silly.var<i32[-2]>
      %1 = arith.constant 0 : i32
      "func.return"(%1) : (i32) -> ()
  }
}
