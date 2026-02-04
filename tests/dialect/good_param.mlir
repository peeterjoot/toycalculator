// RUN: %OptSilly --source %s --out /dev/null

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      // FIXME: there should be a no-params for main verify check for declare.
      %0 = "silly.declare"() <{param_number = 0 : i64, parameter, sym_name = "v"}> : () -> !silly.var<i32>
      %1 = arith.constant 0 : i32
      "silly.return"(%1) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
