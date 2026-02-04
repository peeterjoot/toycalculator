## tag: V3 (Jun 2, 2025)

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

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
