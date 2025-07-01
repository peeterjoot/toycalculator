## TODO

* Function support: WIP:
    - Builder: implement CALL builder.
    - Builder: implement FUNCTION builder (i.e.: within assignment.)
    - Lowering?: implement function call lowering if required.
    - Haven't tested functions with any statements in them.
        - param lookup doesn't work:
            - Enabling stuff in function.toy:plus3: results in error: v not found.
    - See if regressions in:

        function_intret_intparam
        function_intret_void
        function_void_intparm
        function_void_void
        function

        are still there.  Fix and restore to testit enabled list.
    - All the testerrors.sh tests appear to not fail as desired -- check if fixed.

* Switch to CamelCase uniformly.
* Error handling is pschizophrenic, in parser and elsewhere, mix of: assert(), throw, llvm::unreachable, rewriter.notifyMatchFailure, emitError, ...
* grok suggests:

class syntax_error_exception : public exception_with_context
{
public:
    syntax_error_exception(const char *file, int line, const char *func, const std::string &msg)
        : exception_with_context(file, line, func, msg) {}
};

and return_codes specialization:

enum class return_codes : int
{
    success,          // 0
    cannot_open_file, // 1
    semantic_error,   // 2
    syntax_error,     // 3
    unknown_error     // 4
};

(vs. unknown_error which is returned for everything now.)

* string literal tests for edge cases: shortstring.toy: two bugs unresolved.
* array member assignment.
* debug test cases for non-string array variables.  Need array member assignment first.
* NOT operator: add more comprehensive all types testing.
* tests for all the type conversions (i.e.: binary and unary arith operators)
* Lots of cut and paste duplication for type conversion in lowering.cpp -- split out into helper functions.
* EXIT: enforce i8 return type in the MLIR layer (i.e.: actual UNIX shell semantics.) -- currently set to i32 return.
* Implement IF/WHILE/DO/BREAK/CONTINUE statements.
* More complicated expressions.
* CAST operators.
* Allow EXIT at more than the end of program (that restriction is currently enforced in the grammar.)
* Don't have any traits defined for my MLIR operations (initially caused compile errors, and I just commented-out or omitted after that.)
* gdb session for simpleless.toy is not behaving right with respect to 'next'.  Suspect that this is due to my cachine of the one/zero constants, reusing previous location info inappropriately.  Try not caching that and see if it fixes it -- nope.
Breakpoint 1, main () at simpleless.toy:4
4       i1 = TRUE;
(gdb) n
5       j1 = FALSE;
(gdb)
6       PRINT i1;
(gdb)
1
7       PRINT j1;
(gdb)
0
9       b = i1 < j1;
(gdb)
10      PRINT b;
(gdb)
9       b = i1 < j1;
(gdb)
10      PRINT b;
(gdb)
0
12      b = j1 < i1;
(gdb)
13      PRINT b;
(gdb)
12      b = j1 < i1;
(gdb)
13      PRINT b;
(gdb)
1
__libc_start_call_main (main=main@entry=0x400470 <main>, argc=argc@entry=1, argv=argv@entry=0x7fffffffdc08) at ../sysdeps/nptl/libc_start_call_main.h:74
74        exit (result);

-- see the line numbers jump around.

Trickier, but would be fun:
* Implement a JIT so that the "language" has an interpretor mode, as well as static compilation.


