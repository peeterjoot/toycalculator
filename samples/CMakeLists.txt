# Use generator expression for the silly driver so it works in eventual multi-config builds (Debug/Release)
set(SILLY_EXE $<TARGET_FILE:silly>)

# Directory containing your .silly files and expected/*.out
set(SAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_FILES
0minus.silly
a2.silly
addi.silly
andop.silly
array_assign_and_use.silly
array_dynamic_index.silly
array_elem_as_arg.silly
array_in_comparison.silly
array_index_var.silly
array_in_expr_min.silly
array_in_expr.silly
array_lvalue_complex.silly
array_mixed_type.silly
arrayprod.silly
array.silly
assign_to_array_complex.silly
bin.silly
bool_arith_mixed.silly
boolcall0.silly
boolcall1.silly
boolcall2.silly
bool_in_arith.silly
boolliteral.silly
bool_not_on_comparison.silly
boolop.silly
boolr.silly
bool.silly
call_in_predicate.silly
call_with_param_referenced.silly
chained_comparison_parens.silly
const_overflow.silly
conversions.silly
converti.silly
dcl.silly
div_zero_float.silly
elif2.silly
elif3.silly
elif4.silly
elif5.silly
elif6.silly
elif.silly
empty.silly
eqop.silly
error_arrayreturn.silly
error_array_too_many_init.silly
error_chained_comparison2.silly
error_chained_equality.silly
error_expressions4c.silly
error_expressions4d.silly
error_expressions4e.silly
error_induction_var_in_step.silly
error_init_list1.silly
error_init_list2.silly
error_invalid_binary.silly
error_keyword_declare2.silly
error_keyword_declare.silly
error_nested_ivar_conflict.silly
error_notfloat.silly
error_redeclare.silly
error_return_expr_no_return_type.silly
error_shadow_induction.silly
error_stringreturn.silly
error_triple_nested_for_with_shadowing.silly
error_undeclare.silly
exit3.silly
exit42.silly
exitarrayelement.silly
exitx.silly
expressions1.silly
expressions2.silly
expressions3.silly
expressions4a.silly
expressions4b.silly
expressions4.silly
expressions5.silly
expressions6.silly
expressions_assignment.silly
expressions_in_call.silly
expressions_in_for.silly
expressions_in_if.silly
factorial2.silly
factorial.silly
fatal.silly
foo.silly
for_complex_body.silly
for_simplest.silly
forward_ref_init.silly
f.silly
function_add.silly
function_foo_with_local.silly
function_intret_intparam.silly
function_intret_void.silly
function_plist_nocast.silly
function_plist.silly
function_return.silly
function.silly
function_two.silly
function_void_intparm.silly
function_void_void.silly
getboolfail.silly
getbool.silly
get.silly
ifdcl.silly
if.silly
induction_not_used.silly
initarray.silly
init_expr_bool.silly
init_expr_call.silly
init_expr_unary.silly
initlist_param.silly
initlist.silly
initlisttrunc.silly
initsequence.silly
init.silly
inits.silly
initstring.silly
intarray.silly
lessbug2.silly
lessbug3.silly
lessbug.silly
lesseqop.silly
lessop.silly
loadstore.silly
longstring.silly
minimal_eliftest.silly
minmax.silly
mixed_type_assignment.silly
negative_step_for.silly
neg.silly
neqop.silly
nested_calls.silly
nested_for.silly
nested_if_basic.silly
nested_if_deep.silly
nested_if_elif_else.silly
not_on_expr.silly
not.silly
orop.silly
printboollit.silly
printcont.silly
printdi_nested.silly
printdi.silly
printexpr.silly
printlit.silly
print_multiple.silly
printsome.silly
return_complex.silly
return_expression.silly
s2.silly
scopebug.silly
shortstring2.silly
shortstring3.silly
shortstring.silly
simpleless.silly
simplest.silly
stringlit.silly
test.silly
two_declare_no_init.silly
types.silly
unary_on_array_elem.silly
unary.silly
xorop.silly
zero_iter_for.silly
)

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)   # dcl, simplest, ...

    add_test(
        NAME EndToEnd.${test_name}
        COMMAND ${CMAKE_SOURCE_DIR}/bin/testit
                -j
                ${test_file}
        WORKING_DIRECTORY ${SAMPLES_DIR}
    )

    set_tests_properties(EndToEnd.${test_name}
        PROPERTIES
            LABELS "end-to-end;compiler-cycle;sample"
            TIMEOUT 60
            PASS_REGULAR_EXPRESSION "TEST WAS SUCCESSFUL"
            # Show more context on failure
            ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
    )
endforeach()
