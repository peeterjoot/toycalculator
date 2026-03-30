module {
  func.func @main() -> i32 {
    %c3_i64 = arith.constant 3 : i64 loc(#loc3)
    %0 = arith.trunci %c3_i64 : i64 to i32 loc(#loc3)
    %1 = "silly.declare"(%0) : (i32) -> !silly.var<i32> loc(#loc1)
    "silly.debug_name"(%1) <{name = "x"}> : (!silly.var<i32>) -> () loc(#loc1)

    "silly.scope_begin"() <{id = 0 : i32}> : () -> () loc(#loc4)
    %2 = silly.load %1 : <i32> : i32 loc(#loc4)
    %c4_i64 = arith.constant 4 : i64 loc(#loc5)
    %3 = silly.cmp less %2 : i32, %c4_i64 : i64 -> i1 loc(#loc4)

    scf.if %3 {
      "silly.scope_begin"() <{id = 0 : i32}> : () -> () loc(#loc8)

      %c1_i64 = arith.constant 1 : i64 loc(#loc8)
      %4 = silly.load %1 : <i32> : i32 loc(#loc9)
      %5 = silly.arith_binop add %c1_i64 : i64, %4 : i32 -> i64 loc(#loc8)
      %6 = arith.trunci %5 : i64 to i32 loc(#loc8)
      %7 = "silly.declare"(%6) : (i32) -> !silly.var<i32> loc(#loc10)
      "silly.debug_name"(%7) <{name = "myScopeVar"}> : (!silly.var<i32>) -> () loc(#loc10)
      %8 = silly.load %7 : <i32> : i32 loc(#loc11)
      %c0_i32_1 = arith.constant 0 : i32 loc(#loc12)
      "silly.print"(%c0_i32_1, %8) : (i32, i32) -> () loc(#loc12)

      "silly.scope_end"() <{id = 0 : i32}> : () -> () loc(#loc11)

    } else {
    } loc(#loc6)
    "silly.scope_end"() <{id = 0 : i32}> : () -> () loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    return %c0_i32 : i32 loc(#loc2)
  } loc(#loc13)
} loc(#loc)
#loc = loc("if.silly":0:0)
#loc1 = loc("if.silly":1:1)
#loc2 = loc("if.silly":9:1)
#loc3 = loc("if.silly":1:11)
#loc4 = loc("if.silly":3:6)
#loc5 = loc("if.silly":3:10)
#loc6 = loc("if.silly":3:1)
#loc7 = loc("if.silly":4:1)
#loc8 = loc("if.silly":5:23)
#loc9 = loc("if.silly":5:27)
#loc10 = loc("if.silly":5:4)
#loc11 = loc("if.silly":7:10)
#loc12 = loc("if.silly":7:4)
#loc13 = loc(fused[#loc1, #loc2])
