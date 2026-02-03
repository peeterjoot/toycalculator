// RUN: %OptSilly --source %s --out /dev/null

module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      %c1_i64 = arith.constant 1 : i64
      %0 = "silly.declare"(%c1_i64) <{sym_name = "anInitializedScalar"}> : (i64) -> !silly.var<i64>
      %1 = "silly.declare"() <{sym_name = "scalar"}> : () -> !silly.var<i64>
      %2 = "silly.declare"() <{sym_name = "array"}> : () -> !silly.var<i64[10]>
      %c0_i32 = arith.constant 0 : i32
      "silly.return"(%c0_i32) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
