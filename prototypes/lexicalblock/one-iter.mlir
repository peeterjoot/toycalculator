module {
  func.func @main() -> i32 {
    %c0_i64 = arith.constant 0 : i64 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    "silly.print"(%c0_i32, %c0_i64) : (i32, i64) -> () loc(#loc1)

    // FOR header block — induction var declare + init, scope 1
    "silly.scope_begin"() <{id = 1 : i32}> : () -> () loc(#loc4)
    %c1_i64 = arith.constant 1 : i64 loc(#loc5)
    %start = arith.trunci %c1_i64 : i64 to i32 loc(#loc5)
    %c2_i64 = arith.constant 2 : i64 loc(#loc6)
    %end = arith.trunci %c2_i64 : i64 to i32 loc(#loc6)
    %c1_i32 = arith.constant 1 : i32 loc(#loc4)
    %i = "silly.declare"(%start) : (i32) -> !silly.var<i32> loc(#loc7)
    "silly.debug_name"(%i) <{name = "i"}> : (!silly.var<i32>) -> () loc(#loc7)
    "silly.scope_end"() <{id = 1 : i32}> : () -> () loc(#loc11)
    cf.br ^bb1 loc(#loc4)

  ^bb1:  // for_cond — pred: ^bb0, ^bb3
    %i_val_cond = silly.load %i : <i32> : i32 loc(#loc4)
    %cond = arith.cmpi slt, %i_val_cond, %end : i32 loc(#loc4)
    cf.cond_br %cond, ^bb2, ^bb4 loc(#loc4)

  ^bb2:  // for_body — pred: ^bb1
    "silly.scope_begin"() <{id = 2 : i32}> : () -> () loc(#loc8)
    %i_val_body = silly.load %i : <i32> : i32 loc(#loc9)
    %c0_i32_3 = arith.constant 0 : i32 loc(#loc9)
    "silly.print"(%c0_i32_3, %i_val_body) : (i32, i32) -> () loc(#loc9)
    "silly.scope_end"() <{id = 2 : i32}> : () -> () loc(#loc10)
    cf.br ^bb3 loc(#loc10)

  ^bb3:  // for_inc — pred: ^bb2
    %i_val_inc = silly.load %i : <i32> : i32 loc(#loc4)
    %i_next = arith.addi %i_val_inc, %c1_i32 : i32 loc(#loc4)
    silly.assign %i : <i32> = %i_next : i32 loc(#loc4)
    cf.br ^bb1 loc(#loc4)

  ^bb4:  // for_end — pred: ^bb1
    %c2_i64_0 = arith.constant 2 : i64 loc(#loc12)
    %c0_i32_1 = arith.constant 0 : i32 loc(#loc13)
    "silly.print"(%c0_i32_1, %c2_i64_0) : (i32, i64) -> () loc(#loc13)
    %c0_i32_2 = arith.constant 0 : i32 loc(#loc2)
    return %c0_i32_2 : i32 loc(#loc2)
  } loc(#loc14)
} loc(#loc)
#loc = loc("one-iter.silly":0:0)
#loc1 = loc("one-iter.silly":5:5)
#loc2 = loc("one-iter.silly":14:1)
#loc3 = loc("one-iter.silly":5:11)
#loc4 = loc("one-iter.silly":7:5)
#loc5 = loc("one-iter.silly":7:21)
#loc6 = loc("one-iter.silly":7:24)
#loc7 = loc("one-iter.silly":7:17)
#loc8 = loc("one-iter.silly":8:5)
#loc9 = loc("one-iter.silly":9:9)
#loc10 = loc("one-iter.silly":10:5)
#loc11 = loc("one-iter.silly":7:27)
#loc12 = loc("one-iter.silly":12:11)
#loc13 = loc("one-iter.silly":12:5)
#loc14 = loc(fused[#loc1, #loc2])
