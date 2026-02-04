// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 0 : i64
      // CHECK: error: 'silly.declare' op initializer type 'i64' does not match variable element type 'i32'
      %0 = "silly.declare"(%c0, %c0, %c1) <{sym_name = "c"}> : (i32, i32, i64) -> !silly.var<i32[3]>
      %1 = arith.constant 0 : i32
      "silly.return"(%1) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
