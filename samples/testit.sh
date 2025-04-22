set -x
#../build/toycalculator ./unary.toy
#exit
for i in simplest.toy foo.toy test.toy error_redeclare.toy error_unassigned.toy error_undeclare.toy ; do
    echo $i
    cat $i
    ../build/toycalculator --location $i
done

# vim: et ts=4 sw=4
