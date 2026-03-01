## tag: V0 (May 4, 2025)

Language elements:

* Declare a double equivalent variable:

```
  DCL variablename;
```

* Unary assignments to a variable (constants or plus or minus variable-name):

```
  x = 3;
  x = +x;
  x = -x;
```

* Rudimentary binary operations:

```
  x = 5 + 3;
  y = x * 2;
```

* A PRINT builtin.

### MLIR examples

This version of the compiler used the memref dialect.  Example:

```
  > ./build/silly  samples/foo.silly  -g
  "builtin.module"() ({
    "silly.program"() ({
      %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
      "silly.declare"() <{name = "x"}> : () -> () loc(#loc1)
      %1 = "arith.constant"() <{value = 3 : i64}> : () -> i64 loc(#loc2)
      %2 = "silly.unary"(%1) <{op = "+"}> : (i64) -> f64 loc(#loc2)
      "memref.store"(%2, %0) : (f64, memref<f64>) -> () loc(#loc2)
      "silly.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc3)
      "silly.print"(%0) : (memref<f64>) -> () loc(#loc4)
      "silly.return"() : () -> () loc(#loc1)
    }) : () -> () loc(#loc1)
  }) : () -> () loc(#loc)
  #loc = loc("../samples/foo.silly":1:1)
  #loc1 = loc("../samples/foo.silly":1:1)
  #loc2 = loc("../samples/foo.silly":2:5)
  #loc3 = loc("../samples/foo.silly":2:1)
  #loc4 = loc("../samples/foo.silly":4:6)
```

That was removed in V1, which now uses a MLIR dialect that matches the language more closely, deferring alloca to lowering.

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
