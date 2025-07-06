## TODO

* Add integer literal support to PRINT, so that I can do a program as simple as:
    PRINT 42;

* Function support: WIP:
    - Builder: implement FUNCTION builder (i.e.: within assignment.)
    - All the testerrors.sh tests appear to not fail as desired -- still an issue.
    - DI instrumentation isn't right (variable lookup actually works in bar, but foo shows up with the module location)

            Breakpoint 1, main () at function_plist.toy:19
            19      PRINT "In main";
            (gdb) n
            In main
            20      CALL foo();
            (gdb) s
            foo () at function_plist.toy:1
            1       FUNCTION bar ( INT16 w, INT32 z )
            (gdb) n
            12          v = 3;
            (gdb) p w
            No symbol "w" in current context.
            (gdb) n
            13          PRINT "In foo";
            (gdb) p w
            No symbol "w" in current context.
            (gdb) n
            In foo
            14          CALL bar( v, 42 );
            (gdb) p v
            $1 = 3
            (gdb) s
            bar (w=64, z=0) at function_plist.toy:1
            1       FUNCTION bar ( INT16 w, INT32 z )
            (gdb) n
            3           PRINT "In bar";
            (gdb) n
            In bar
            4           PRINT w;
            (gdb) bt
            #0  bar (w=3, z=42) at function_plist.toy:4
            #1  0x00000000004004a6 in foo () at function_plist.toy:14
            #2  0x0000000000400505 in main () at function_plist.toy:20
            (gdb) p w
            $2 = 3


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


