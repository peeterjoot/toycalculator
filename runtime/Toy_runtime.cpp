/**
 * @file    Toy_runtime.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Runtime functions for the toy compiler language.
 */
#include <stdio.h>

extern "C"
void __toy_print( double value )
{
    printf( "%f\n", value );
}
