# arrayprod -- requires: --init-fill 255
# 'initlist' => '--init-fill 255',
# 'initarray' => '--init-fill 255',
set(INITFILL_TESTS
    arrayprod
    initarray
    initlist
)

add_endtoend_tests("EndToEnd.complex.array" ${INITFILL_TESTS})
