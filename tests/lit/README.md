# LIT tests

Example to run a LIT suite from the repo TOP dir:
```
    $HOME/build-llvm/bin/llvm-lit -v tests/lit/debug/
```

or just one test:
```
    $HOME/build-llvm/bin/llvm-lit -v tests/lit/debug/two-declare-order-location.silly
```

In that test dir, can run without any path:

```
    $HOME/build-llvm/bin/llvm-lit -v two-declare-order-location.silly
```
