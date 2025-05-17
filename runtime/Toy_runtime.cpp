/**
 * @file    Toy_runtime.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Runtime functions for the toy compiler language.
 */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

extern "C"
{
    void __toy_print_f64( double value )
    {
        printf( "%f\n", value );
    }

    void __toy_print_f32( float value )
    {
        printf( "%f\n", value );
    }

    void __toy_print_i64( int64_t value )
    {
        printf( "%" PRId64 "\n", value );
    }

    void __toy_print_i32( int32_t value )
    {
        printf( "%d\n", value );
    }

    void __toy_print_i16( int16_t value )
    {
        printf( "%hd\n", value );
    }

    void __toy_print_i8( int8_t value )
    {
        printf( "%d\n", (int)value );
    }
}
