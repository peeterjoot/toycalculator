// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      // was trying to trigger "only 1D arrays are supported" error in declare verify, but the type parser emits an error first:
      // : expected ']'
      // CHECK: error: array-size must be followed immediately by ]
      %0 = "silly.declare"() : () -> !silly.var<i32[7,2]>
      %1 = arith.constant 0 : i32
      "silly.return"(%1) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
