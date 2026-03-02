# Debug-info LIT tests

These tests validate source location attribution, variable scoping, `debug_name` ops, etc.

Run from the repo TOP dir with:
```
    $HOME/build-llvm/bin/llvm-lit -v tests/lit/debug/
```
or just one:
```
    $HOME/build-llvm/bin/llvm-lit -v tests/lit/debug/debug-two-declare-order.silly
```

In this dir, run one with:

```
    $HOME/build-llvm/bin/llvm-lit -v debug-two-declare-order.silly
```
