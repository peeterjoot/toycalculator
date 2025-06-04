## tag: V4 (WIP)

INTERNALS:
* Generate the __toy_print... prototypes on demand, to clutter up the generated code less.  Can do this by saving and restoring the insertion point to the module level (where the symbol table and globals live.)

## tag: V3

LANGUAGE ELEMENTS:
* comparison operators (<, <=, EQ, NE) yielding BOOL values.  These work for any combinations of floating and integer types (including BOOL.)
* integer bitwise operators (OR, AND, XOR).  These only for for integer types (including BOOL.)
* a NOT operator, yielding BOOL.
* Array + string declaration and lowering support, including debug instrumentation, and print support for string variables.
* String assignment support.

TEST:
* move samples/testit.sh to bin/testit
* testit: Document --optimize.  Add --assembly, --no-debug
* test case generators for all the boolean and bitwise operations.
* many new tests.

INTERNALS:
* Fixed -g/-OX propagation to lowering.  If -g not specified, now don't generate the DI.
* Show the optimized .ll with --emit-llvm instead of the just-lowered .ll (unless not invoking the assembly printer, where the ll optimization passes are registered.)
* Reorganize the grammar so that all the simple lexer tokens are last.  Rename a bunch of the tokens, introducing some consistency.
* calculator.td: introduce IntOrFloat constraint type, replacing AnyType usage; array decl support, and string support.
* driver: writeLL helper function, pass -g to lowering if set.
* parser: handle large integer constants properly, array decl support, and string support.
* simplest.cpp: This MWE is updated to include a global variable and global variable access.
* parser: implicit exit: use the last saved location, instead of the module start.  This means the line numbers don't jump around at the very end of the program anymore (i.e.: implicit return/exit)

## tag: V2

This release:

* Adds DWARF debug instrumentation.  Yay!
* Builds a link step into the compiler driver.  To avoid that, compile with -c.

## tag: V1

* Declare variables with BOOL, INT8, INT16, INT32, INT64, FLOAT32, FLOAT64 types:
```
BOOL b;
INT16 i;
FLOAT32 f;
```
* TRUE, FALSE, and floating point constants:
```
b = TRUE;
f = 5 + 3.14E0;
```
* An EXIT builtin to return a Unix command line value (must be the last statement in the program):
```
EXIT 1;
EXIT x;
```
* Expression type conversions:
```
INT16 x;
FLOAT32 y;
y = 3.14E0;
x = 1 + y;
```
The type conversion rules in the language are not like C.
Instead, all expression elements are converted to the type of the destination before the operation, and integers are truncated.
Example:
```
INT32 x;
x = 1.78 + 3.86E0;
FLOAT64 f;
f = x;
PRINT f;
f = 1.78 + 3.86E0;
PRINT f;
```

The expected output for this program is:
```
4.000000
5.640000
```

### MLIR examples

The MLIR for the language now matches the statements of the language much more closely.  Consider test.toy for example:

```
DCL x;
x = 5 + 3.14E0;
PRINT x;
DCL y;
y = x * 2;
PRINT y;
```

for which the MLIR is now free of memref dialect:

```
"builtin.module"() ({
  "toy.program"() ({
    "toy.declare"() <{name = "x", type = f64}> : () -> () loc(#loc)
    %0 = "arith.constant"() <{value = 5 : i64}> : () -> i64 loc(#loc1)
    %1 = "arith.constant"() <{value = 3.140000e+00 : f64}> : () -> f64 loc(#loc1)
    %2 = "toy.add"(%0, %1) : (i64, f64) -> f64 loc(#loc1)
    "toy.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc1)
    %3 = "toy.load"() <{name = "x"}> : () -> f64 loc(#loc2)
    "toy.print"(%3) : (f64) -> () loc(#loc2)
    "toy.declare"() <{name = "y", type = f64}> : () -> () loc(#loc3)
    %4 = "toy.load"() <{name = "x"}> : () -> f64 loc(#loc4)
    %5 = "arith.constant"() <{value = 2 : i64}> : () -> i64 loc(#loc4)
    %6 = "toy.mul"(%4, %5) : (f64, i64) -> f64 loc(#loc4)
    "toy.assign"(%6) <{name = "y"}> : (f64) -> () loc(#loc4)
    %7 = "toy.load"() <{name = "y"}> : () -> f64 loc(#loc5)
    "toy.print"(%7) : (f64) -> () loc(#loc5)
    "toy.exit"() : () -> () loc(#loc)
  }) : () -> () loc(#loc)
}) : () -> () loc(#loc)
#loc = loc("test.toy":1:1)
#loc1 = loc("test.toy":2:5)
#loc2 = loc("test.toy":3:1)
#loc3 = loc("test.toy":4:1)
#loc4 = loc("test.toy":5:5)
#loc5 = loc("test.toy":6:1)
```

I'm still using llvm.alloca, but that now doesn't show up until lowering:
```
; ModuleID = 'test.toy'
source_filename = "test.toy"

declare void @__toy_print(double)

define i32 @main() {
  %1 = alloca double, i64 1, align 8
  store double 8.140000e+00, ptr %1, align 8
  %2 = load double, ptr %1, align 8
  call void @__toy_print(double %2)
  %3 = alloca double, i64 1, align 8
  %4 = load double, ptr %1, align 8
  %5 = fmul double %4, 2.000000e+00
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  call void @__toy_print(double %6)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

Example of the generated assembly code for this program:
```
0000000000000000 <main>:
   0:   push   %rax
   1:   movsd  0x0(%rip),%xmm0        # 9 <main+0x9>
   9:   call   e <main+0xe>
   e:   movsd  0x0(%rip),%xmm0        # 16 <main+0x16>
  16:   call   1b <main+0x1b>
  1b:   xor    %eax,%eax
  1d:   pop    %rcx
  1e:   ret
```

## tag: V0

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
> ./build/toycalculator  samples/foo.toy  -g
"builtin.module"() ({
  "toy.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "toy.declare"() <{name = "x"}> : () -> () loc(#loc1)
    %1 = "arith.constant"() <{value = 3 : i64}> : () -> i64 loc(#loc2)
    %2 = "toy.unary"(%1) <{op = "+"}> : (i64) -> f64 loc(#loc2)
    "memref.store"(%2, %0) : (f64, memref<f64>) -> () loc(#loc2)
    "toy.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc3)
    "toy.print"(%0) : (memref<f64>) -> () loc(#loc4)
    "toy.return"() : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc("../samples/foo.toy":1:1)
#loc1 = loc("../samples/foo.toy":1:1)
#loc2 = loc("../samples/foo.toy":2:5)
#loc3 = loc("../samples/foo.toy":2:1)
#loc4 = loc("../samples/foo.toy":4:6)
```

That was removed in V1, which now uses a MLIR dialect that matches the language more closely, deferring alloca to lowering.
