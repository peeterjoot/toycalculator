set -x

for i in \
simplest.toy \
foo.toy \
test.toy \
error_redeclare.toy \
error_unassigned.toy \
error_undeclare.toy \
error_invalid_binary.toy \
error_invalid_unary.toy \
    ; do
    echo $i
    cat $i
    ../build/toycalculator --location $i
done

# vim: et ts=4 sw=4
