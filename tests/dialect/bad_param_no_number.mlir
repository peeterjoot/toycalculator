// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      // CHECK: error: 'silly.declare' op parameter declarations require a 'param_number' attribute.
      %0 = "silly.declare"() <{parameter, sym_name = "v"}> : () -> !silly.var<i32>
      %1 = arith.constant 0 : i32
      "silly.return"(%1) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
