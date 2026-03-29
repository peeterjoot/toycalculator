# Changelog V0.11.0 (WIP)

## Bug fix:

```
INT64 y = 3;
BOOL lt2 = (y < 10);
```

Had inappropriate i1 narrowing for the literal creation in bowls of parseExpression (both front ends.)

## Bison FE.

* Fixed all the non-debug/syntax-error testsuite failures.

## Debugging.

* Updated the dwarf types to use the silly-language type names.
