# RCs:
#   'getboolfail' => '134', # abort
#
# INPUT:
#   'get' => '6 7 1 2 3 4 5',
#   'getbool' => '0 1'
#   'getboolfail' => '2',
set(GET_TESTS
    get
    getboolfail
    getbool
)

add_endtoend_tests("EndToEnd.get" ${GET_TESTS})
