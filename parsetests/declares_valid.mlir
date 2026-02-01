// RUN: mlir-opt-silly --source %s --out /dev/null

module {
  func.func @test() {
    %c1_i64 = arith.constant 1 : i64
    %0 = "silly.declare"(%c1_i64) <{sym_name = "anInitializedScalar"}> : (i64) -> !silly.var<i64>
    %0 = "silly.declare"() <{sym_name = "scalar"}> : () -> !silly.var<i64>
    %1 = "silly.declare"() <{sym_name = "array"}> : () -> !silly.var<i64[10]>
  }
}
