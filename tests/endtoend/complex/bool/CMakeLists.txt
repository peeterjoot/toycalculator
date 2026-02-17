# RCs:
#   'bool' => 1,
#   'boolr' => 1,
#   'getboolfail' => '134', # abort
#
# INPUT:
#   'getbool' => '0 1'
#   'getboolfail' => '2',
set(COMPLEX_BOOL_TESTS
    boolr
    bool
    getboolfail
    getbool
)

add_endtoend_tests("EndToEnd.complex.bool" ${COMPLEX_BOOL_TESTS})
