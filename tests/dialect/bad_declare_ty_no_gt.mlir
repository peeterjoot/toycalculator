// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      // CHECK: error: unbalanced '<' character in pretty dialect name
      %0 = "silly.declare"() <{sym_name = "t"}> : () -> !silly.var<i32
      %1 = arith.constant 0 : i32
      "silly.return"(%1) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
