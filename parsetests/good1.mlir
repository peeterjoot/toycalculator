%c1_i64 = arith.constant 1 : i64
%0 = "silly.declare"(%c1_i64) <{sym_name = "anInitializedScalar"}> : (i64) -> !silly.var<i64>
