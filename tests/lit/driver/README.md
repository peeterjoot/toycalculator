To run just one test (noisily), use something like:

$HOME/build-llvm/bin/llvm-lit -v $HOME/toycalculator/tests/lit/driver/the-test-name.silly

or the whole driver suite:

$HOME/build-llvm/bin/llvm-lit -v $HOME/toycalculator/tests/lit/driver
