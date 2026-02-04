// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      // Was trying to trigger verify error: "result must be of type !silly.var" -- but think that tablegen enforces the
      // constraint before the verify even runs.

      // CHECK: error: 'silly.declare' op result #0 must be Abstract variable location (scalar or array), but got 'i32'
      %0 = "silly.declare"() <{sym_name = "v"}> : () -> i32
      %1 = arith.constant 0 : i32
      "silly.return"(%1) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
